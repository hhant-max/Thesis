import java.util.*;

/**
 * <p>Title: Graph </p>
 * <p>Description: Reprenting of bipatite Graph </p>
 * <p>Copyright: Copyright (c) 2002 Zhihua </p>
 * <p>Company: York Univ </p>
 * @author Zhihua
 * @version 1.0
 */

/* vertex in a Bipartite graph */
class Vertex{
  private final boolean UNMATCHED = false;
  private final boolean MATCHED = true;
  private final boolean LEFT = false;
  private final boolean RIGHT = true;
  boolean mathced = false;
  boolean isLeft = false;
  int outdegree = 0;
  int indegree = 0;
}

/* the graph is reprented in adjacent list */
public class graph {

  /* we use a vector to represent a edge list in a directed graph
     for example, adjacentList[1] has 2 means there is a edge from
  point 1 to point 2*/

  Vector adjacentList[];
  /* vertex list */
  Vertex vertex[];
  /* this list is used to store the augmenting Path we got from matching */
  Vector augmentPath;
  /* total number of all points,points in left side and right side */
  int points, leftpoints, rightpoints;

  /* create the graph,points means the toal number of points */
  public graph(int points, int leftpoints, int rightpoints) {
    this.leftpoints = leftpoints;
    this.rightpoints = rightpoints;
    this.points = points;

    adjacentList = new Vector[points];
    vertex = new Vertex[points];
    augmentPath = new Vector();

    for (int i = 0; i < points; i++){
      vertex[i] = new Vertex();
      if ( i < leftpoints) vertex[i].isLeft = true;
      adjacentList[i] = new Vector();
    }

  }
  /* add edge, add an edge to the graph */
  public void addedge(int startPoint,int endPoint){
  /* insert the edge to the adjacentList of startPoint */
    adjacentList[startPoint].addElement(new Integer(endPoint));
    /* increase the outdegree of startPoint, indegree of endPoint */
    vertex[startPoint].outdegree += 1;
    vertex[endPoint].indegree += 1;
    /* if the edge is from right to left side, mark both start and end as mached point */
    if (isRight(startPoint)&& isLeft(endPoint)){
      vertex[startPoint].mathced = true;
      vertex[endPoint].mathced = true;

    }
  }

  /* remove edge */
  public void removeedge(int startPoint,int endPoint){
   /* find the index of edge in the adjacentList of startPoint */
   int index = adjacentList[startPoint].indexOf(new Integer(endPoint));
   /* remove the edge from adjacentList of startPoint */
   if ( index > - 1) adjacentList[startPoint].removeElementAt(index);
   /* decrease the outdegree of startPoint and indegree of endPoint */
   vertex[startPoint].outdegree -= 1;
   vertex[endPoint].indegree -= 1;

   /* if the startPoint is on the right, and its outdegree become zero,
      mark the startPoint as unmached */
   if (isRight(startPoint) && vertex[startPoint].outdegree == 0)
     vertex[startPoint].mathced = false;

   /* if the endPoint is on the left, and its indegree become zero,
      mark the endPoint as unmached */
   if (isLeft(endPoint) && vertex[endPoint].indegree == 0)
     vertex[endPoint].mathced = false;

  }

  /* reverse edge */
  /* change the direction of edge, change i to j to j to i */
  public void reverseedge(int startPoint, int endPoint){
    removeedge(startPoint,endPoint);
    addedge(endPoint,startPoint);
  }
  /* xor */
  /* reverse all the edges in the augmenting path*/
  public String XOR(boolean showstatus){
    int start, end;
    String str = "";
    /* the first point of augmenting path */
    start = ((Integer)augmentPath.elementAt(0)).intValue();

    if (showstatus)str += "The augument Path is \n";

    for (int i = 1; i< augmentPath.size(); i++){
      end = ((Integer)augmentPath.elementAt(i)).intValue();

      if (showstatus){
        /* this part is used to show the augmenting path, just for test */
        String prtStr = "";
        prtStr += start < leftpoints ? "A"+(start+1) : "G"+(start+1-leftpoints);
        prtStr += " to ";
        prtStr += end < leftpoints ? "A"+(end+1) : "G"+(end+1-leftpoints);
        str += prtStr + "\n";
      }

      reverseedge(start,end); /* reverse the edge */
      start = end;
    }
    return str;
  }
  /* do the maximium bipartiture matching */
  public String Matching(boolean showstatus){

    String str = "";
    while ( findAugmentPath()){
      str += XOR(showstatus);
    }
    return str;
  }
  public boolean findAugmentPath(){
    augmentPath.removeAllElements(); /*init the path */
    /* use all the unmatched left points as start, see if we can find
    a augmenting path */
    for ( int i = 0; i < leftpoints; i++){
      if ( vertex[i].mathced == false){
        if ( findPath(i))
          return true;
        else
          augmentPath.removeAllElements(); /*re init the path */
        }
    }
    return false;
  }
  /* recursive find a path using DFS */
  public boolean findPath(int start){
    int nextPt,index;
    /* if the current vertex has no out edge, return false */
    if (vertex[start].outdegree == 0) return false;
    /* insert the current point to the path */
    augmentPath.addElement(new Integer(start));

    /* use the pts that the current point is linked to as next point,
    recursively call findPath function */
    for ( int i = 0; i < adjacentList[start].size(); i++){
      nextPt = ((Integer)adjacentList[start].elementAt(i)).intValue();
      /*if the next point was already in the path, discard it */
      if ( augmentPath.indexOf(new Integer(nextPt)) > -1) continue;
      /* find a terminal, add it to the path and return true */
      if ( vertex[nextPt].mathced == false){
        augmentPath.addElement(new Integer(nextPt));
        return true;
        }
      /* otherwise recursive call using depth first search */
      else if ( findPath(nextPt))
        return true;

    }
    /* if failed, delete the current pt from path and return false */
    index = augmentPath.indexOf(new Integer(start));
    augmentPath.removeElementAt(index);
    return false;

  }


  /* indicate whether the current point is in right side */
  public boolean isLeft(int pt){
    if ( pt < leftpoints) return true;
    else return false;
  }
  /* indicate whether the current point is in right side */
  public boolean isRight(int pt){
    if ( pt > leftpoints - 1) return true;
    else return false;
  }

  /* print out the current status of the graph */
  public String toString(){
    String str = "";
    for ( int i = 0; i < points; i++){
      str += "Point ";
      str += isLeft(i) == true ? "A" + (i+1):"G" + (i-leftpoints+1);
      str += " is ";
      str += vertex[i].mathced == true ? "MATCHED\n" : "UNMATCHED\n";
      for (int j = 0; j < adjacentList[i].size(); j++){
        int to = ((Integer)adjacentList[i].elementAt(j)).intValue();
        str += " and is connected to points ";
        str += isLeft(to) ? "A" + (to+1) : "G" + (to-leftpoints+1);
        str += "\n";
      }
    }
    return str;
  }

}
