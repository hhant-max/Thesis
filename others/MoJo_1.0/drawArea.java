import java.io.*;
import java.util.*;
import javax.swing.*;

/**
 * <p>Title:  </p>
 * <p>Description: The canvas of mojo applet </p>
 * <p>Copyright: Copyright (c) 2002</p>
 * <p>Company: </p>
 * @author unascribed
 * @version 1.0
 */
import java.awt.*;

public class drawArea extends Canvas{
  /* the y position for the first cluster */
  protected final int space = 15;
  int totalCost = 0;
  int delay = 100; /*time of delay */
  mojoAnimation controller;
  boolean showGraph = false;
  boolean showGroupTag = false;
  boolean isGroupAssigned = false;
  GraphAni graph = null;
  private Image offScreenImage;  // 2nd buffer
  private Graphics offScreenGraphics, canvasGraphics; // main screen and 2nd buffer's graphics context
  private final int cheight = 150;
  private final int leftCluster = 1;
  private final int rightCluster = 0;
  int l = 0;    /* number of clusters in A */
  int m = 0;    /* number of clusters in B */
  int o = 0;    /* number of objects in system */
  Vector tempA, tempB;
  ClusterAni A[];    /* A */
  ClusterAni B[];    /* B */
  Obj O[]; /* object list */

  /* record the capicity of each group, if the group is empty
  ,the count is zero, otherwise >= 1 */
  int groupscount[];

  int moves = 0; /* total number of move operations */
  int no_of_nonempty_group = 0; /* number of total noneempty groups */
  /* after join operations, each group will have only one cluster left,
  we use grouptags[i] to indicate the remain cluster in group i*/

  ClusterAni grouptags[]; /* every none empty group have a tag point to a cluser in A */

  public drawArea(mojoAnimation controller,Vector tempA, Vector tempB, long o) {
    super();
    this.controller = controller;
    this.o = (int)o;
    this.tempA = tempA;
    this.tempB = tempB;
    l = tempA.size();
    m = tempB.size();
    createClusters();
  }

  public drawArea(mojoAnimation controller){
    super();
    this.controller = controller;

  }

  public void reset(){
    showGroupTag = false;
    showGraph = false;
    A = null; B = null; O = null; graph = null;
    l = 0; m = 0;
    totalCost = 0;
    groupscount = null;
    grouptags = null;
    isGroupAssigned = false;
    offScreenGraphics.clearRect(0, 0, getSize().width,getSize().height);
    controller.stateLabel.setText("");
    repaint();
    if ( !controller.isRandom ) createClusters();

  }
  public void createBackBuffer(){
    canvasGraphics = getGraphics();
    offScreenImage = createImage(getSize().width, getSize().height);
    if (offScreenGraphics != null){
      offScreenGraphics.dispose();
    }
    offScreenGraphics = offScreenImage.getGraphics();

  }
  public void createClusters(){
    totalCost = 0;
    l = tempA.size();
    m = tempB.size();

    controller.fieldA.setText(""+l);
    controller.fieldB.setText(""+m);
    controller.fieldO.setText(""+o);

    A = new ClusterAni[l]; /* create A */
    B = new ClusterAni[m]; /* create B */
    O = new Obj[o]; /* create O */
    groupscount = new int[m];
    grouptags = new ClusterAni[m];
    /* init group tags */
    for (int j = 0; j<m; j++){
      grouptags[j] = null;
    }
    /* create each cluster in A */
    for (int i = 0; i<l; i++){
      A[i] = new ClusterAni(this,leftCluster,i,l,m);
    }
    /* create each cluster in B */
    for (int i = 0; i<m; i++){
      B[i] = new ClusterAni(this,rightCluster,i,m,l);
    }

    controller.stateLabel.setText("The content of cluster A & B");
    addAllObjects();
    repaint();


  }
  /* output the join operations */
  public void join(){
    showGraph = false; /* hide the graph*/
    /* find none empty groups and find total number of moves */
    for (int i = 0; i<l; i++){
      /* caculate the count of nonempty groups */
      if ( groupscount[A[i].getGroup()] == 0){
        no_of_nonempty_group += 1;
      }
      /* assign group tags */
      if (grouptags[A[i].getGroup()] == null){
        grouptags[A[i].getGroup()] = A[i];
        }
      /* assign the group count */
      groupscount[A[i].getGroup()] += 1;
      /* calculate the number of move opts for each cluster */
      moves += A[i].gettotalTags() - A[i].getMaxtag();
    }


    /* output the join opt */
    String printstr = "";
    for (int j = 0; j<m; j++){
      if ( groupscount[j] > 1 ){ /* the group has at least two clusters */
        totalCost += groupscount[j] - 1;
        printstr = "join ";
        for (int i = 0; i<l; i++){
          if (A[i].getGroup() == j){
            A[i].isjoin = true;
            printstr += "A"+(i+1)+" ";
          }
        }
        printstr += "together Total Cost is " + totalCost;
        controller.stateLabel.setText(printstr);
        draw();
        canvasGraphics.drawImage(offScreenImage,0,0,this);
        try {Thread.sleep(10 * delay );} catch (Exception e) {}; // sleep to control the speed of the animation
        for (int i = 0; i<l; i++){
          if (A[i].getGroup() == j){
            if ( grouptags[j].getNo() != i ) {
              grouptags[j].join(A[i]);
            }

          }
        }
        grouptags[j].isjoin = false;
        draw();
        canvasGraphics.drawImage(offScreenImage,0,0,this);
        try {Thread.sleep(10 * delay);} catch (Exception e) {}; // sleep to control the speed of the animation
        showGroupTag = true;
        repaint();
      }
    }

  }

  /* output the move opt */
  public void move(){
    String printstr = "";
    int newClusterIndex = l; /* index for newly created cluster */

      for (int j = 0; j < m; j++){
        if ( grouptags[j] != null ){
          for ( int i = 0; i <m; i ++ ){
            printstr = "Move ";
            if ( i != j && grouptags[j].tags[i] > 0){

              for ( int k = 0; k < grouptags[j].objectList.size(); k++){
                if (((Integer)grouptags[j].objectgroupList.elementAt(k)).intValue() == i ){
                  totalCost++;
                  printstr += ((Obj)grouptags[j].objectList.elementAt(k)).getName()+" ";
                }
              }

              if ( grouptags[i] != null ) /* the group is not empty */
              {
                printstr += " to A"+(grouptags[i].getNo()+1);
              }
              else
              {
                grouptags[i] = new ClusterAni(this, leftCluster,newClusterIndex++,l,m); /* create a new Group */
                printstr += " to newly created A"+(grouptags[i].getNo()+1)+"(G"+(i+1)+")";
              }
              printstr += " Total Cost is " + totalCost;

              int y = space;
              int oldy = 0;
              int oldheight = 0;
              controller.stateLabel.setText(printstr);
              grouptags[j].setMovingStatus(i,true);
              draw();
              canvasGraphics.drawImage(offScreenImage,0,0,this);
              try {Thread.sleep(10 * delay);} catch (Exception e) {}; // sleep to control the speed of the animation

              grouptags[j].move(i,grouptags[i]);
              draw();
              canvasGraphics.drawImage(offScreenImage,0,0,this);
              try {Thread.sleep(10 * delay);} catch (Exception e) {}; // sleep to control the speed of the animation
              grouptags[i].setMovingStatus(i,false);

            }
          }
        }
      }
      offScreenGraphics.setColor(Color.lightGray);
      offScreenGraphics.clearRect(0, 0, getSize().width,getSize().height);
      showGroupTag = true;
      controller.stateLabel.setText("The total cost of MoJo is "+ totalCost);
      repaint();

  }

  public void Matching(){
    while ( graph.findAugmentPath()){
      graph.drawAugPath(offScreenGraphics);
      canvasGraphics.drawImage(offScreenImage,0,0,this);
      try {Thread.sleep(10 * delay);} catch (Exception e) {}; // sleep to control the speed of the animation
      graph.XOR();
      graph.paint(offScreenGraphics);
      canvasGraphics.drawImage(offScreenImage,0,0,this);
      try {Thread.sleep(10 * delay);} catch (Exception e) {}; // sleep to control the speed of the animation


    }
  }
  public void generateGraph(){
    graph = new GraphAni(this,A,B);
    for ( int i = 0; i < l; i++){
      for (int j = 0; j < A[i].groupList.size(); j ++ ){
        graph.addedge(i,l+((Integer)A[i].groupList.elementAt(j)).intValue());
      }
    }
    controller.stateLabel.setText("Generate the graph according to the Tags");
    showGraph = true;
    repaint();

  }
  public void assignGroup(){
    controller.stateLabel.setText("");
    for ( int i = l; i < l + m; i ++){
      if ( graph.vertex[i].matched ){
        int index = ((Integer)graph.adjacentList[i].elementAt(0)).intValue();
        A[index].setGroup(i-l);
      }
    }

    isGroupAssigned = true;
    repaint();

  }
  public boolean calculateTag(){
    Hashtable tableB = new Hashtable();
    for ( int i = 0; i < m; i++){
      for ( int j = 0; j < B[i].objectList.size(); j ++ ){
        tableB.put(B[i].objectList.elementAt(j),new Integer(i));
      }
    }
    for ( int i = 0; i < l; i++){
      for ( int j = 0; j < A[i].objectList.size(); j ++ ){
        try{
          A[i].addtag( (Integer) tableB.get(A[i].objectList.elementAt(j)));
        }
        catch (NullPointerException e){
          JOptionPane.showMessageDialog(this,"Error: object "+A[i].objectList.elementAt(j)+"can not be found in target partition" );
          return false;
        }
      }
    }
    controller.stateLabel.setText("Assigning the tags");
    repaint();
    return true;
  }
  /* random set all the objects in clusters, and make sure each cluster has
  at least one objects */
  public void addAllObjects(){
    int objnumber = 0;
    for (int i = 0; i < tempA.size(); i++){
      for (int j = 0; j < ((Vector)tempA.elementAt(i)).size(); j ++ ){
        O[objnumber] = new Obj((String)((Vector)tempA.elementAt(i)).elementAt(j));
        A[i].addobject(O[objnumber]);
        objnumber++;
      }
    }

    for (int i = 0; i < tempB.size(); i++){
      for (int j = 0; j < ((Vector)tempB.elementAt(i)).size(); j ++ ){
        String objName = (String)((Vector)tempB.elementAt(i)).elementAt(j);
        for (int k = 0; k < objnumber; k ++){
          if (objName.equals(O[k].getName()))
              B[i].addobject(O[k]);
        }
      }
    }
  }

  public void paint(Graphics g) {
    if (offScreenGraphics == null){
      createBackBuffer();
      g.clearRect(0, 0, getSize().width,getSize().height);
    }
    draw();
    g.drawImage(offScreenImage,0,0,this);
  }
  public void draw(){
    offScreenGraphics.clearRect(0, 0, getSize().width,getSize().height);
    int y = space;
    int oldy = 0;
    int oldheight = 0;
    if (showGroupTag && grouptags != null){
      for ( int i = 0; i < m; i++){
      if (grouptags[i] != null){
        if ( i != 0) y = oldy + oldheight + space;
        grouptags[i].paint(offScreenGraphics,y);
        oldy = y;
        oldheight = grouptags[i].getHeight();
      }
      }
    }
    else{
      if ( A != null){
        for ( int i = 0; i < l; i++){
          if ( i != 0) y = oldy + oldheight+space;
          if (A[i] != null){
            A[i].paint(offScreenGraphics,y);
            oldy = y;
            oldheight = A[i].getHeight();
          }
        }
      }
    }
    y = space;
    for ( int j = 0; j < m; j++){
        if ( j != 0) y = oldy + oldheight+space;
        B[j].paint(offScreenGraphics,y);
        oldy = y;
        oldheight = B[j].getHeight();
    }
    if (showGraph) {
      graph.paint(offScreenGraphics);
    }
  }

  public void randomCreate(){
    l = Integer.parseInt(controller.fieldA.getText());
    m = Integer.parseInt(controller.fieldB.getText());
    o = Integer.parseInt(controller.fieldO.getText());

    A = new ClusterAni[l]; /* create A */
    B = new ClusterAni[m]; /* create B */
    O = new Obj[o]; /* create O */
    groupscount = new int[m];
    grouptags = new ClusterAni[m];
    /* init group tags */
    for (int j = 0; j<m; j++){
      grouptags[j] = null;
    }
    /* create each cluster in A */
    for (int i = 0; i<l; i++){
      A[i] = new ClusterAni(this,leftCluster,i,l,m);
    }
    /* create each cluster in B */
    for (int i = 0; i<m; i++){
      B[i] = new ClusterAni(this,rightCluster,i,m,l);
    }
    /* crate each object */
    for (int i = 0; i<o; i++){
      O[i] = new Obj("b"+(i+1));
    }
    randomSet(A,O);
    randomSet(B,O);
    repaint();
  }
  /* random set all the objects in clusters, and make sure each cluster has
  at least one objects */
  public void randomSet(ClusterAni c[],Obj obj[]){

    int csize = c.length;
    int osize = obj.length;
    int emptyNumber = csize;
    int index;

    for ( int i = 0; i < osize; i++){
      index = (int)(Math.random()*csize);
      if ( emptyNumber < osize - i ){
        if (c[index].isempty = true) emptyNumber--;
        c[index].addobject(obj[i]);
      }
      else {
        while (c[index].isempty == false){
          index = (int)(Math.random()*csize);
          }
        emptyNumber--;
        c[index].addobject(obj[i]);
      }
    }

  }

/*  public void update(Graphics g) {
    paint(g);
  }*/


}
