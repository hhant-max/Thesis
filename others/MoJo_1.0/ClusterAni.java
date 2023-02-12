import java.awt.*;
import java.util.*;
/**
 * <p>Title: cluster </p>
 * <p>Description: </p>
 * <p>Copyright: Zhihua Wen Copyright (c) 2002</p>
 * <p>Company: York University </p>
 * @author Zhihua Wen
 * @version 1.0
 */

public class ClusterAni {


  protected final int objperrow = 10; /* how many objects can each row contain */
  protected final int objwidth = 30; /* the width of each object */
  protected final int objheight = 30; /* the height of each object */
  protected final int xoffset = 10; /* the xoffset of the first object in cluster */
  protected final int yoffset = 10; /* the yoffset of the first object in cluster */
  protected final int space = 15; /* the space bewteen each row in cluser */

  /* whether the currrent cluster is in left or right side */
  private final int leftCluster = 1;
  private final int rightCluster = 0;

  /* the x postion for cluster in A and B */
  private final int leftOfA = 40;
  private final int leftOfB = 440;
  /* the y position for the first cluster in A or B*/
  protected final int top = 20;
  /* the width of cluster */
  private final int width = 310;
  /* the height of cluster, it can be changed to fit the number of objects */
  private int height = 0;
  /* position of Top left */
  int x,y;

  /* parent controll */
  drawArea parent;
  /* is joining with others */
  boolean isjoin = false;
  /* is in left side */
  boolean isleft = true;
  /* is empty */
  boolean isempty = true;
  /* the serial Number of this  A */
  private int no = 0;
  /* no. of cluster in A */
  private int l = 0;
  /* no. of cluster in B */
  private int m = 0;
  /* max V(ij), the maximium tags */
  private int maxtag = 0;
  /* |Ai|, total objects in Ai */
  private int totaltags = 0;
  /* the total number of total groups */
  private int groupNo = 0;
  /* the group No that Ai belongs to */
  private int group = 0;
  /*number of each tags */
  protected int tags[];
  /* object list */
  public Vector objectList;
  /* indicate the group of each object */
  public Vector objectgroupList;
  /* group list */
  public Vector groupList;


  public int getWidth(){
  	return width;
  }
  public int getHeight(){
  	return height;
  }
  public int getNo(){
    return no;
  }

  public int getGroup(){
    return group;
  }
  public void setGroup(int group){
    this.group = group;
  }
  public int getMaxtag(){
    return maxtag;
  }
  public int gettotalTags(){
    return totaltags;
  }

  public ClusterAni() {
  }
  public ClusterAni(drawArea parent,int side, int no, int l,int m){
    this.parent = parent;

    if (side == leftCluster)
      this.isleft = true;
    if (side == rightCluster)
      this.isleft = false;

    isempty = true;
    isjoin = false;
    this.l = l;
    this.m = m;
    this.no = no;
    tags = new int[m];
    objectList = new Vector();
    objectgroupList = new Vector();
    groupList = new Vector();
    for (int j = 0; j<m; j++){
      tags[j] = 0;
    }
  }
  public void addobject(Obj object){
    isempty = false;
    totaltags += 1;
    objectList.addElement(object);

  }
  public void minustag(Integer t){
    int tag = t.intValue();
    totaltags -= 1;
    if (totaltags == 0) isempty = true;
     tags[tag] -= 1;
  }
  public int addtag(Integer t){
    int tag = t.intValue();
    if (tag >=0 && tag < m){
      isempty = false;
      objectgroupList.addElement(t);
      tags[tag] += 1;

      /* if tags is max & unique,then change group to it & clear grouplist */
      if ( tags[tag] > maxtag){
        maxtag = tags[tag];
        group = tag;
        groupNo = 1;
        groupList.removeAllElements();
        groupList.addElement(t);
      }
      /* if tags is max but not nuique,then add it to the grouplist */
      else
      if ( tags[tag] == maxtag){
        groupNo += 1;
        groupList.addElement(t);
      }
    }
    return group;
  }

  /* move objects to another cluster */
  public void move(int grouptag,ClusterAni sub){
    int newx,newy,oldx,oldy;
    Obj tempobj;
    int i = 0;
    while (i < objectList.size()){
      if ( ((Integer)objectgroupList.elementAt(i)).intValue() == grouptag){
        tempobj = (Obj)objectList.elementAt(i);
        setObjPos(i);
        oldx = tempobj.getX();
        oldy = tempobj.getY();
        sub.objectList.addElement(tempobj);
        sub.addtag(new Integer(grouptag));
        sub.setObjPos(sub.objectgroupList.size()-1);
        newx = tempobj.getX();
        newy = tempobj.getY();
        //parent.drawMoveObject(tempobj,oldx,oldy,newx,newy);
        objectList.removeElementAt(i);
        objectgroupList.removeElementAt(i);
        minustag(new Integer(grouptag));

      }else i++;


    }
  }

  /* join with another cluster */
  public void join(ClusterAni sub){
      for ( int i = 0; i < sub.objectList.size(); i++){
        addobject((Obj)sub.objectList.elementAt(i));
        addtag((Integer)(sub.objectgroupList.elementAt(i)));
      }
    sub.isempty = true;
    sub.objectgroupList.removeAllElements();
    sub.objectList.removeAllElements();
    sub = null;

  }
  /* set moving status,this method is only used for drawing,set all the ojbects
     in a group to is moving status, then we can draw it in red colors */
  public void setMovingStatus(int grouptag,boolean isMoving){

   for ( int i = 0; i < objectList.size(); i++){
      if ( ((Integer)objectgroupList.elementAt(i)).intValue() == grouptag){
        ((Obj)objectList.elementAt(i)).isMoving = isMoving;
      };
   }

  }
  /* set object position */
  public void setObjPos(int index){
  	Obj tempobj = ((Obj)objectList.elementAt(index));
  	tempobj.setPos(x+xoffset+objwidth*(index%objperrow),y+yoffset+objheight*(index/objperrow));
  }

  /* draw method */
  public void paint(Graphics g, int y ){

    if (isempty){
      this.height = 0;
      return;
    }
    /* if the cluster is joining with others,show it in red colors */
    if (isjoin && isleft ) g.setColor(Color.red);
    else g.setColor(Color.black);
    /* if the current cluster is in left side, nmae start with "A",otherwise "B" */
    String drawStr = isleft == true ? "A"+(no+1):"B"+(no+1);

    /* if the tag assigment was already done, the objectgroupList will be none empty,show the tags */
    if (objectgroupList.size() > 0 ){
      for ( int i = 0; i<m; i++){
        if ( tags[i] != 0 ) drawStr += " T"+(i+1)+":"+tags[i];
      }
    }
    /* if the group selection was already done, show the group number */
    if (parent.isGroupAssigned && this.isleft) {
      drawStr += " Group G"+(group+1);
    };

    /* choose the x position of the current cluster, if it's in left,start with left of A, otherwise left of B */
    x = isleft == true ? leftOfA : leftOfB;
    /* set the y position */
    this.y = y;
    /* adjust height according the total number of objects in it */
    height = (int)Math.ceil( (double)objectList.size() / (double)objperrow ) * objheight+space;


    g.drawString(drawStr,x,y); /* draw the caption string, such as A1: 2 T2 3 T3 G3 */
    g.drawRect(x,y,width,height); /* draw the retangle rect which represent the cluser */
    /* drawing each objects */
    for ( int i = 0; i < objectList.size(); i++){
      setObjPos(i);
      ((Obj)objectList.elementAt(i)).paint(this,g);
    }
    g.setColor(Color.black);
  }
}
