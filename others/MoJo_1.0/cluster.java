import java.util.*;
/**
 * <p>Title: cluster </p>
 * <p>Description: </p>
 * <p>Copyright: Zhihua Wen Copyright (c) 2002</p>
 * <p>Company: York University </p>
 * @author Zhihua Wen
 * @version 1.0
 */

public class cluster {

  /* is empty */
  boolean isempty;
  /* No.  A */
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
  /* tags */
  private int tags[];
  /* total misplaced omnipresent objects */
  private int misplacedOmnipresentObjects = 0;
  /* object list */
  public Vector objectList[];
  /* group list */
  public Vector groupList;


  public int getMisplacedOmnipresentObjects(){
    return misplacedOmnipresentObjects;
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
  public int getGroupNo(){
    return groupNo;
  };
  public void setGroupNo(int groupNo){
    this.groupNo = groupNo;
  }
  public void minusGroupNo(){
    groupNo--;
  }
  public cluster() {
  }
  public cluster(int no, int l,int m){
    isempty = false;
    this.l = l;
    this.m = m;
    this.no = no;
    tags = new int[m];
    objectList = new Vector[m];
    groupList = new Vector();
    for (int j = 0; j<m; j++){
      tags[j] = 0;
      objectList[j] = new Vector();
    }
  }
  public int addobject_mojoplus(int t, String object){
    if ( t >= 0 && t<m){
      if ( tags[t] == 0 ){
        group = t;
        groupNo += 1;
        tags[t] = 1;
        totaltags += 1;
        groupList.addElement(new Integer(t));
        objectList[t].addElement(object);
        maxtag = 1;
      }

    }
    return t;
  }
  public int addobject(int t,String object){
    if (t>=0 && t<m){
      tags[t] += 1;
      totaltags += 1;
      objectList[t].addElement(object);

      /* if tags is max & unique,then change group to it & clear grouplist */
      if ( tags[t] > maxtag){
        maxtag = tags[t];
        group = t;
        groupNo = 1;
        groupList.removeAllElements();
        groupList.addElement(new Integer(t));
      }
      /* if tags is max but not nuique,then add it to the grouplist */
      else
      if ( tags[t] == maxtag){
        groupNo += 1;
        groupList.addElement(new Integer(t));
      }
    }
    return group;
  }
  public String toString(){
    String str ="";
    str = str + "A" + (no+1) + " is in group G" + (group+1)+"\n";

    for (int i = 0; i<m; i++){
      if (objectList[i].size() > 0)
      {
         str = str + "Group "+(i+1)+":"+" have " + objectList[i].size()+" objects, they are "+objectList[i].toString()+"\n";

      }

    }
    return str;
  }

  /* move objects to another cluster */
  public void move(int grouptag,cluster sub){
    for ( int i = 0; i < objectList[grouptag].size(); i++){
      sub.objectList[grouptag].addElement(objectList[grouptag].elementAt(i));
    }
    objectList[grouptag].removeAllElements();
  }

  /* merge with another cluster */
  public void merge(cluster sub){
    maxtag += sub.maxtag;
    totaltags += sub.totaltags;
    sub.isempty = true;
    for (int j = 0; j<m ; j++){
      for ( int i = 0; i < sub.objectList[j].size(); i++)
        objectList[j].addElement(sub.objectList[j].elementAt(i));

    }

  }
  /* detect whether is a omnipresent object */
  private boolean isOmniPresent(String obj, Vector omniVector){
    if ( omniVector == null || omniVector.size() == 0 ) return false;
    if (omniVector.indexOf(obj) != - 1)
      return true;
    else
      return false;
  }
  /* edge metric cost */
  /* calculation abs(edges(obj_j,A_i)-edges(obj_i,A_j)/ edges(obj_j,A_i)+edges(obj_i,A_j)*/
  /* We assume we will move obj_j from cluster A_i to cluster A_j*/
  public double edgeCost(Hashtable tableR,cluster[] grouptags, Vector omniVector){
    double cost = 0;
    double value = 0, value1 = 0, value2 = 0;
    for (int j=0; j<m; j++){
      if (j != group){
        if ( grouptags[j] == null) {
          /* if target cluster is null, i.e. we need to create one
             then the additional edge cost is abs(edge(A_i,obj_j)-0)/edge(A_i,obj_j)+0 = 1
             we don't need to calculate edge(A_i,obj_j) anymore */
          //cost += objectList[j].size();
          /* search omnipresent objects in objectList[j], T_j in A_i */
          for (int i = 0; i < objectList[j].size(); i++){
            String obj = (String)objectList[j].elementAt(i);
            if (isOmniPresent(obj,omniVector))
              misplacedOmnipresentObjects++;
            else
              cost++;
          }
        }
        else{
          for (int i = 0; i < objectList[j].size(); i++){
            String obj = (String)objectList[j].elementAt(i);
            if (isOmniPresent(obj,omniVector))misplacedOmnipresentObjects++;
            else{
              double edges_source = 0,edges_target = 0;
              /*calculate edges(A_i,obj_j),calculate only total connections between obj_j and all the tag T_i in A_i*/
              for ( int k = 0; k < objectList[group].size(); k++){
                String obj2 = (String)objectList[group].elementAt(k);
                /* we consider both the connection from obj to obj2 and obj2 to obj as same, so we plus them together as
                the total connection strength between obj and obj2 */
                value1 = tableR.get((String)(obj+"%@$"+obj2))== null ? 0: ((Double)tableR.get(obj+"%@$"+obj2)).doubleValue();
                value2 = tableR.get((String)(obj2+"%@$"+obj))== null ? 0: ((Double)tableR.get(obj2+"%@$"+obj)).doubleValue();
                edges_source += value1 + value2;
              }
              /*calculate edges(A_j,obj_j),calculate only total connections between obj_j and all the tag T_j in A_j*/
              for (int l = 0; l < grouptags[j].objectList[j].size(); l++){
                String obj3 = (String)grouptags[j].objectList[j].elementAt(l);
                value1 = tableR.get((String)(obj+"%@$"+obj3))== null ? 0: ((Double)tableR.get(obj+"%@$"+obj3)).doubleValue();
                value2 = tableR.get((String)(obj3+"%@$"+obj))== null ? 0: ((Double)tableR.get(obj3+"%@$"+obj)).doubleValue();
                edges_target += value1+value2;
              }
              if ( edges_target != edges_source)
                cost += Math.abs(edges_source-edges_target)/(edges_source+edges_target);
            }
          }
        }
      }
    }
    return cost;
  }

}
