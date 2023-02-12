import java.awt.*;
import java.util.*;

/**
 * <p>Title: GraphAni </p>
 * <p>Description: Reprenting of bipatite GraphAni </p>
 * <p>Copyright: Copyright (c) 2002 Zhihua </p>
 * <p>Company: York Univ </p>
 * @author Zhihua
 * @version 1.0
 */

/* vertex in a Bipartite graph */

/* the graph is reprented in adjacent list */
public class GraphAni {

  private final boolean UNMATCHED = false;
  private final boolean MATCHED = true;
  private final boolean LEFT = false;
  private final boolean RIGHT = true;
  /* we use a vector to represent a edge list in a directed graph
     for example, adjacentList[1] has 2 means there is a edge from
  point 1 to point 2*/

  drawArea parent;
  Vector adjacentList[];
  /* vertex list */
  VertexAni vertex[];
  /* this list is used to store the augmenting Path we got from matching */
  Vector augmentPath;
  /* total number of all points,points in left side and right side */
  int points, leftpoints, rightpoints;

  /* create the graph,points means the toal number of points */
  public GraphAni(drawArea parent,ClusterAni A[], ClusterAni B[]) {
    this.parent = parent;
    this.leftpoints = A.length;
    this.rightpoints = B.length;
    this.points = leftpoints + rightpoints;

    adjacentList = new Vector[points];
    vertex = new VertexAni[points];
    augmentPath = new Vector();

    for (int i = 0; i < points; i++){
      if ( i < leftpoints){
        vertex[i] = new VertexAni(A[i]);
      }
      else {
        vertex[i] = new VertexAni(B[i-leftpoints]);
      }

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
      vertex[startPoint].matched = true;
      vertex[endPoint].matched = true;

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
     vertex[startPoint].matched = false;

   /* if the endPoint is on the left, and its indegree become zero,
      mark the endPoint as unmached */
   if (isLeft(endPoint) && vertex[endPoint].indegree == 0)
     vertex[endPoint].matched = false;

  }

  /* reverse edge */
  /* change the direction of edge, change i to j to j to i */
  public void reverseedge(int startPoint, int endPoint){
    removeedge(startPoint,endPoint);
    addedge(endPoint,startPoint);
  }
  /* xor */
  /* reverse all the edges in the augmenting path*/
  public void XOR(){
    int start, end;
    /* the first point of augmenting path */
    start = ((Integer)augmentPath.elementAt(0)).intValue();
    String prtStr = "";
    prtStr += start < leftpoints ? "A"+(start+1) : "G"+(start+1-leftpoints);

    //System.out.println("The augument Path is ");

    for (int i = 1; i< augmentPath.size(); i++){
      end = ((Integer)augmentPath.elementAt(i)).intValue();

      /* this part is used to show the augmenting path, just for test */
      prtStr += " to ";
      prtStr += end < leftpoints ? "A"+(end+1) : "G"+(end+1-leftpoints);

      System.out.println(prtStr);

      reverseedge(start,end); /* reverse the edge */
      start = end;
    }
    parent.controller.stateLabel.setText("The augument Path is from "+ prtStr);
  }
  /* do the maximium bipartiture matching */
  public void Matching(){

    while ( findAugmentPath()){
      XOR();
    }
  }
  public boolean findAugmentPath(){
    augmentPath.removeAllElements(); /*init the path */
    /* use all the unmatched left points as start, see if we can find
    a augmenting path */
    for ( int i = 0; i < leftpoints; i++){
      if ( vertex[i].matched == false){
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
      if ( vertex[nextPt].matched == false && isRight(nextPt) ){
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

  /* draw the agumenting path */
  public void drawAugPath(Graphics g){
    g.setColor(Color.green);
    int start,end;
    if  ( augmentPath.size() < 2 ) return;
    start = ((Integer)augmentPath.elementAt(0)).intValue();
    for ( int i = 1; i < augmentPath.size(); i++){
      end = ((Integer)augmentPath.elementAt(i)).intValue();
      drawLink(g,start,end);
      start = end;
    }
    g.setColor(Color.black);
  }
  /* draw the graph */
  public void paint(Graphics g){
    for ( int i = 0; i < points; i++){
      vertex[i].paint(g);
      for (int j = 0; j < adjacentList[i].size(); j++){
        int to = ((Integer)adjacentList[i].elementAt(j)).intValue();
        if (isLeft(i) && isRight(to)) g.setColor(Color.black);
        if (isRight(i) && isLeft(to)) g.setColor(Color.red);
        drawLink(g,i,to);
        g.setColor(Color.black);
      }
    }

  }

  /* draw the edge between two points */
  public void drawLink(Graphics g,int from, int to)
  {

    int x1 = vertex[from].x;
    int y1 = vertex[from].y;
    int x2 = vertex[to].x;
    int y2 = vertex[to].y;
    g.drawLine(x1,y1,x2,y2);
  }

  /* print out the current status of the graph */
  public String toString(){
    String str = "";
    for ( int i = 0; i < points; i++){
      str += "Point ";
      str += isLeft(i) == true ? "A" + (i+1):"G" + (i-leftpoints+1);
      str += " is ";
      str += vertex[i].matched == true ? "MATCHED\n" : "UNMATCHED\n";
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
class VertexAni{
  private final boolean UNMATCHED = false;
  private final boolean MATCHED = true;
  private final boolean LEFT = false;
  private final boolean RIGHT = true;
  boolean matched = false;
  boolean isLeft = false;
  int outdegree = 0;
  int indegree = 0;
  int x,y;
  public VertexAni(ClusterAni cluster){
    this.isLeft = cluster.isleft;
    if (isLeft){
      this.x = cluster.x + cluster.getWidth();
      this.y = cluster.y + cluster.getHeight() / 2;
    }
    else {
      this.x = cluster.x;
      this.y = cluster.y + cluster.getHeight() / 2;
    }

  }
  public void paint(Graphics g){
    if (matched) g.setColor(Color.red);
    else g.setColor(Color.black);
    g.drawOval(x-5,y-5,5,5);
    g.setColor(Color.black);
  }
}
