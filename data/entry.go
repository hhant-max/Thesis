package raft

import (
	"sort"
)

type LogEntry struct {
	Term int
	Command interface{}
}


type AppendEntriesArgs struct {
	Term         int
	LeaderId     int
	prevLogIndex int
	prevLogTerm  int
	Entries      []LogEntry
	LeaderCommit int
}

// AppendEntriesReply AppendEntries函数的RPC回复参数
type AppendEntriesReply struct {
	CurrentTerm    int  // currentTerm, for leader to update itself
	Success        bool // true if follower contained entry matching prevLogIndex and prevLogTerm
	ConflictTerm   int
	FirstIndex     int
}

func max(a, b int) int {
	if a > b {
		return a
	}

	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}




func (rf *Raft) wakeupConsistencyCheck() {
	for i := 0;i < len(rf.peers);i++ {
		if i != rf.me {
			rf.newEntryCond[i].Broadcast()
		}
	}
}

func (rf * Raft) logEntryAgreeDaemon() {
	for i := 0;i < len(rf.peers);i++ {
		if i != rf.me {
			go rf.consistencyCheckDaemon(i)
		}
	}
}

func (rf *Raft)consistencyCheckDaemon(n int) {
	for {
		rf.mu.Lock()
		rf.newEntryCond[n].Wait()

		select {
		case <- rf.shutdown:
			rf.mu.Unlock()
			return
		default:
		}

		if rf.isLeader {
			var args AppendEntriesArgs
			args.Term = rf.CurrentTerm
			args.LeaderId = rf.me
			args.LeaderCommit = rf.commitIndex
			args.prevLogIndex = rf.nextIndex[n]-1
			args.prevLogTerm = rf.Logs[args.prevLogIndex].Term

			if rf.nextIndex[n] < len(rf.Logs) {
				args.Entries = append(args.Entries,rf.Logs[rf.nextIndex[n]:]...)
			} else {
				args.Entries = nil
			}

			rf.mu.Unlock()

			replyCh := make(chan AppendEntriesReply,1)
			go func() {
				var reply AppendEntriesReply
				if rf.sendAppendEntries(n,&args,&reply) {
					replyCh <- reply
				}
			}()

			select {
			case reply:= <-replyCh:
				rf.mu.Lock()
				if reply.Success {
					rf.matchIndex[n] = reply.FirstIndex
					rf.nextIndex[n] = rf.matchIndex[n] + 1
					rf.updateCommitIndex()
				} else {
					if reply.CurrentTerm > args.Term {
						if reply.CurrentTerm > rf.CurrentTerm {
							rf.CurrentTerm = reply.CurrentTerm
							rf.VotedFor = -1
						}

						if rf.isLeader {
							rf.isLeader = false
							rf.wakeupConsistencyCheck()
						}

						rf.persist()
						rf.mu.Unlock()
						rf.resetTimer <- struct{}{}
						return
					}
					var know, lastIndex = false, 0
					if reply.ConflictTerm != 0 {
						for i := len(rf.Logs) - 1;i > 0;i-- {
							if rf.Logs[i].Term == reply.ConflictTerm {
								know = true
								lastIndex = i
								break
							}
						}

						if know {
							if lastIndex > reply.FirstIndex {
								lastIndex = reply.FirstIndex
							}

							rf.nextIndex[n] = lastIndex
						} else {
							rf.nextIndex[n] = reply.FirstIndex
						}
					} else {
						rf.nextIndex[n] = reply.FirstIndex
					}

					rf.nextIndex[n] = min(max(rf.nextIndex[n],1),len(rf.Logs))
				}

				rf.mu.Unlock()
			}
		} else {
			rf.mu.Unlock()
			return
		}
	}
}


func (rf *Raft) sendAppendEntries(server int, args *AppendEntriesArgs, reply *AppendEntriesReply) bool {
	ok := rf.peers[server].Call("Raft.AppendEntries", args, reply)
	return ok
}



func (rf *Raft) applyEntryDaemon() {
	for {
		var logs []LogEntry
		rf.mu.Lock()
		for rf.lastApplied == rf.commitIndex {
			rf.commitCond.Wait()
			select {
				case <-rf.shutdown:
					rf.mu.Unlock()
					close(rf.applyCh)
					return
			default:
			}
		}

		last,cur := rf.lastApplied,rf.commitIndex
		if last < cur {
			rf.lastApplied = rf.commitIndex
			logs = make([]LogEntry,cur - last)
			copy(logs,rf.Logs[last + 1:cur + 1])
		}

		rf.mu.Unlock()

		for i:= 0;i < cur-last;i++ {
			reply := ApplyMsg {
				Index :last + i + 1,
				Command: logs[i].Command,
			}

			rf.applyCh <- reply
		}
	}
}

func (rf *Raft) updateCommitIndex() {
	match := make([]int,len(rf.matchIndex))
	copy(match, rf.matchIndex)

	sort.Ints(match)
	target := match[len(rf.peers) / 2]

	if rf.commitIndex < target {
		if rf.Logs[target].Term == rf.CurrentTerm {
			rf.commitIndex = target
			go func() {rf.commitCond.Broadcast()}()
		}
	}
}

func (rf *Raft) AppendEntries(args *AppendEntriesArgs, reply *AppendEntriesReply) {
	select {
		case <-rf.shutdown:
			return
		default:
	}

	rf.mu.Lock()
	defer rf.mu.Unlock()

	if args.Term < rf.CurrentTerm {
		reply.CurrentTerm = rf.CurrentTerm
		reply.Success = false
		return
	}

	if rf.isLeader {
		rf.isLeader = false
		rf.wakeupConsistencyCheck()
	}

	if rf.VotedFor != args.LeaderId {
		rf.VotedFor = args.LeaderId
	}

	if rf.CurrentTerm < args.Term {
		rf.CurrentTerm = args.Term
	}

	rf.resetTimer <- struct{}{}

	preLogIdx, preLogTerm := 0, 0

	if len(rf.Logs) > args.prevLogIndex {
		preLogIdx = args.prevLogIndex
		preLogTerm = rf.Logs[preLogIdx].Term
	}

	if preLogIdx == args.prevLogIndex && preLogTerm == args.prevLogTerm {
		reply.Success = true
		rf.Logs = rf.Logs[:preLogIdx+1]
		rf.Logs = append(rf.Logs,args.Entries...)

		var last = len(rf.Logs) - 1

		if args.LeaderCommit > rf.commitIndex {
			rf.commitIndex = min(args.LeaderCommit,last)

			go func() {rf.commitCond.Broadcast()}()
		}

		reply.ConflictTerm = rf.Logs[last].Term
		reply.FirstIndex = last
	} else {
		reply.Success = false

		var first = 1
		reply.ConflictTerm = preLogTerm
		if reply.ConflictTerm == 0 {
			first = len(rf.Logs)
			reply.ConflictTerm = rf.Logs[first-1].Term
		} else {
			for i := preLogIdx-1;i > 0;i-- {
				if rf.Logs[i].Term != preLogTerm {
					first = i + 1
					break
				}
			}

		}

		reply.FirstIndex = first
	}

	rf.persist()
}