/*modified at 12:15 Wednesday, march 15th */
import java.io.*;
import java.util.*;
/**
 * <p>Title: EMoJo </p>
 * <p>Description: Main window </p>
 * <p>Copyright: Copyright (c) 2002 Zhihua Wen</p>
 * <p>Company: York Univ</p>
 * @author Zhihua Wen
 * @version 1.0.6
 */

public class MoJoCommand {


    /* we show the process of MoJo */
    static final int showStatus = 1;
    /* we calculate the MoJo metric */
    static final int mojoMetric = 2;
    /* we use mojoPlus*/
    static final int mojoPlus = 4;
    /*we use edge MoJo */
    static final int edgeMoJo = 8;
    /* none special action */
    static final int normalMoJo = 0;

    /* the initial action */
    static int action = normalMoJo;

    /* this hashtable is used to store the relation between object and cluster in B */
    static Hashtable tableObj = new Hashtable();
    /* countA & countB is used to calculate the different number of clusters in A & B */
    static Hashtable tableA = new Hashtable();
    static Hashtable tableB = new Hashtable();
    /*relation between objects */
    static Hashtable tableR = new Hashtable();
    /*file name */
    static String sourceFile = "",targetFile = "", relationFile = "";
    /* use for store the name of each items */
    static Vector nameA = new Vector();
    static Vector nameB = new Vector();
    /* use for store the number of ojbects in each cluster in partition B
       this is use in calculate the maxdistance from partition B */
    static Vector numberB = new Vector();
    /* This vector is used to store all the objects and cluster in A */
    static Vector tempA  = new Vector();


    static int l = 0;    /* number of clusters in A */
    static int m = 0;    /* number of clusters in B */
    static long o = 0;   /* number of total objects */
    static long totalCost = 0; /*total cost of Mojo */
    static cluster A[] = null;    /* A */

    /* record the capicity of each group, if the group is empty
    ,the count is zero, otherwise >= 1 */
    static int groupscount[] = null;

    /* after join operations, each group will have only one cluster left,
    we use grouptags[i] to indicate the remain cluster in group i*/
    static cluster grouptags[] = null; /* every none empty group have a tag point to a cluser in A */

    /*if we want to caculate both the distance between both A and B, then the */
    /* loopNum should be 2, otherwise should be 1 */
    static int loopNum = 1;


    /* init the files */

    static BufferedReader br_t; /* used for reading from target file */
    static BufferedReader br_s; /* used for reading from traget file */
    static BufferedReader br_r; /* used for reading from relation file */

  public static void main(String[] args) {



    if ( args.length != 2 && args.length != 3 && args.length != 4)
    {
      showerrormsg();
    };
    /* there are two running option
       if we use java a.rsf b.rsf directly
       then sourceFile = args[0] = a.rsf = the first argument
       targetFile = args[1] = b.rsf  the second argument

       if we use java argument a.rsf b.rsf, then we have 3 arguments
       args[0] = parameter = first argument
       then sourceFile = args[1] = a.rsf = the second argument
       targetFile = args[2] = b.rsf  the third argument
    */

    /* two arguments situation, in this case, we calculate both a to b and b to a, then loopNum = 2 */
    switch (args.length){
      case 2 :
      targetFile = args[1];
      sourceFile = args[0];
      loopNum = 2;
      break;


    /* three arugments situation, in this case, the action depends on the parameter */
    case 3:
      targetFile = args[2];
      sourceFile = args[1];

      /* -as holds for single direction MoJoPlus */
      if ( args[0].equalsIgnoreCase("-as"))
      {
        action = mojoPlus;
        loopNum = 1;
      }
      else
      /* -a holds for double direction MoJoPlus */
      if ( args[0].equalsIgnoreCase("-a"))
      {
        action = mojoPlus;
        loopNum = 2;
      }
      else
      /* -s holds for single direction MoJo */
      if ( args[0].equalsIgnoreCase("-s"))
      {
        loopNum = 1;
      }
      else
      /* -v means view the single direction mojo process */
      if ( args[0].equalsIgnoreCase("-v"))
      {
        action = showStatus;
        loopNum = 1;
      }
      else
      /* -m means to see the metric of MoJo distance */
      if ( args[0].equalsIgnoreCase("-m"))
      {
        action = mojoMetric;
        loopNum = 1;
      }
      else
      {
        showerrormsg();
      };
    break;
    /*argument = 4, only the case of edge MoJo */
    case 4:
    if (args[0].equalsIgnoreCase("-e"))
    {
      targetFile = args[2];
      sourceFile = args[1];
      relationFile = args[3];
      action = edgeMoJo;
      loopNum = 1;
    }
    else
    {
      showerrormsg();
    }
    break;
    default:
    break;
    };

    for ( int loop = 0; loop < loopNum; loop++){
      /* if it's edge mojo mode, we try to open the relationship file first */
      if (args.length == 4){
        readRelationRSFfile();
      };


      /*read target file */
      if (isBunch(targetFile))
        readTargetBunchFile();
      else
        readTargetRSFFile();
      /*read source file */
      if (isBunch(sourceFile))
        readSourceBunchFile();
      else
        readSourceRSFfile();

      l = tableA.size(); /* number of clusters in A */
      m = tableB.size(); /* number of clusters in B */
      if ( o != tableObj.size())
      {
        System.out.println("The total number of ojbects in source file does not equal the total number of objects in target file");
        System.exit(0);
      }; /* number of total objects */


      A = new cluster[l]; /* create A */
      groupscount = new int[m]; /* the count of each group, 0 if empty */
      grouptags = new cluster[m]; /* the first cluster in each group, null if empty */

      /* init group tags */
      for (int j = 0; j<m; j++){
        grouptags[j] = null;
      }


      /* create each cluster in A */
      for (int i = 0; i<l; i++){
        A[i] = new cluster(i,l,m);
      }

      /* tag assigment */
      tagAssigment();

      /*optimization*/
      //optimization();

      /* draw graph and matching */
      maxbipartiteMatching();

      /* output the total cost */
      if (loop == 0 ){
        totalCost = calculateCost();
        if (action == mojoMetric){
          /* show the mojo metric distance */
          System.out.println("The MoJo distance metric is " + mojoMetricValue(numberB,o,totalCost) + "%");
        }
      }
      else{
        System.out.println("The Mojo value is "+Math.min(totalCost,calculateCost()));
      };

      /*perform the edgeMoJo */
      if (action == edgeMoJo){
        /*peroform join operation first */
        for (int j = 0; j<m; j++){
          if ( groupscount[j] > 1 ){
            for (int i = 0; i<l; i++){
              if (A[i].getGroup() == j){
                if ( grouptags[j].getNo() != i ) {
                  grouptags[j].merge(A[i]);
                };
              }
            }
          }
        }
        /* calculate the additional edge cost */
        double edgeCost = 0;
        for (int j = 0; j<m; j++){
          if ( grouptags[j] != null )
            edgeCost += grouptags[j].edgeCost(tableR,grouptags,null);
        }
        System.out.println("The additional cost of edge is "+edgeCost);


      }

      /* in practical use, only the above part is useful,
      because we only want to know the minmium number of join + move,
      we dont need to know the exact process, anyway, the following parts
      show how it works */
      if(action == showStatus){
        /* output the merge opt */
        System.out.println("The process of join operations ");
        for (int j = 0; j<m; j++){
          if ( groupscount[j] > 1 ){
            for (int i = 0; i<l; i++){
              if (A[i].getGroup() == j){
                if ( grouptags[j].getNo() != i ) {
                  grouptags[j].merge(A[i]);
                  System.out.println("join clusters "+ (String)nameA.elementAt(grouptags[j].getNo()) +" and "+(String)nameA.elementAt(i));
                };
              }
            }
          }
        }

        /* output the move opt */
        System.out.println("The process of move operations ");
        int newClusterIndex = l; /* index for newly created cluster */
        for (int j = 0; j < m; j++){
          if ( grouptags[j] != null ){
            for ( int i = 0; i <m; i ++ ){
              if ( i != j && grouptags[j].objectList[i].size() > 0){
                System.out.print("Move "+grouptags[j].objectList[i] + " from A"+(grouptags[j].getNo()+1));

                if ( grouptags[i] != null ) /* the group is not empty */
                {
                  System.out.println(" to cluster A"+(grouptags[i].getNo()+1));
                }
                else
                {
                  grouptags[i] = new cluster(newClusterIndex++,l,m); /* create a new Group */
                  System.out.println(" to newly created cluster A"+(grouptags[i].getNo()+1)+"(G"+(i+1)+")");
                }
                grouptags[j].move(i,grouptags[i]);
              }
            }
          }
        }
      }
      /* re init the variable */
      if ( loopNum == 2 && loop == 0){
        /* this hashtable is used to store the relation between object and cluster in B */
        tableObj = new Hashtable();
        /* countA & countB is used to calculate the different number of clusters in A & B */
        tableA = new Hashtable();
        tableB = new Hashtable();
        /*change file name */
        /* the format is MoJo [argument] A B
           in the first loop A = sourceFile B = targetFile
           in the second loop change A = targetFile B = sourceFile */
        if (args.length == 2 ){
	        sourceFile = args[1];
	        targetFile = args[0];
        }
        else
        {
	        sourceFile = args[2];
	        targetFile = args[1];
        };


        /* the following part is for the reinit before second loop
           it was used when we want to calculate both mojo distance from
           A to B and B to A */
        /* use for store the name of each items */
        nameA = new Vector();
        nameB = new Vector();

        /* This vector is used to store all the objects and cluster in A */
        tempA  = new Vector();

        l = 0;    /* number of clusters in A */
        m = 0;    /* number of clusters in B */
        o = 0;   /* number of total objects */
        A = null;    /* A */

        /* record the capicity of each group, if the group is empty
        ,the count is zero, otherwise >= 1 */
        groupscount = null;

        grouptags = null; /* every none empty group have a tag point to a cluser in A */
      };

    }
   }

   static void maxbipartiteMatching(){
      /* create the graph and add all the edges */
      graph bgraph = new graph(l+m,l,m);

      for ( int i = 0; i < l; i++){
         for (int j = 0; j < A[i].groupList.size(); j ++ ){
            bgraph.addedge(i,l+((Integer)A[i].groupList.elementAt(j)).intValue());
         }
      }

      /* use max bipature to caculate the group */
      System.out.println(bgraph.Matching(action == showStatus));
      /* assign group after matching, for each Ai in matching, assign the corresponding group,
      for other cluster in A, just leave them alone */
      for ( int i = l; i < l + m; i ++){
         if ( bgraph.vertex[i].mathced ){
            int index = ((Integer)bgraph.adjacentList[i].elementAt(0)).intValue();
            A[index].setGroup(i-l);
            if (action == showStatus) System.out.println("Assign A"+(index+1)+" to group G"+(i-l+1));
         }
       }

   }

   /* calculate the mojo metirc distance value, using the formula
   Q(M) = 1 - mno(A,B)/ max(mno(any_A,B)) * 100% */
   static double mojoMetricValue(Vector number_of_B, long obj_number,long totalCost){
     long maxDis = maxDistanceTo(number_of_B,obj_number);
     return Math.rint(( 1 - (double)totalCost / (double)maxDis)*10000) / 100;
   }
   /* calculate the max(mno(B, any_A)), which is also the max(mno(any_A, B)) */
   static long maxDistanceTo(Vector number_of_B, long obj_number) {
     int group_number = 0;
     int[] B = new int[number_of_B.size()];

     for (int i = 0; i < B.length; i++ ){
       B[i] = ((Integer)number_of_B.elementAt(i)).intValue();
     }
     /* sort the array in ascending order */
     java.util.Arrays.sort(B);

     for (int i = 0; i < B.length; i++ ){
     /* calculate the minimum maximum possible groups for partition B */
     /* after sort the B_i in ascending order
     	B_i: 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10, 15 we can calculate g in this way
          g: 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 10, 11  */
       if ( group_number < B[i]) group_number++;
     }
     /* return n - l + l - g = n - g */
     return obj_number - group_number;



   }

   static void optimization(){
       /* optimization */
       /* this function is unnecessary. but it can accelerate the process of mojo */
       /* we believe that most cluster in A has only one unique group to select, so we
       can assign the group to them and kick them out. thus few clusters and edges was
       left in graph */

      for ( int i = 0; i < l; i++){
         if ( A[i].getGroupNo() == 1){
            A[i].groupList.removeAllElements();
            A[i].minusGroupNo();
            for ( int j = 0; j <l; j++){
               if ( j != i && A[j].getGroupNo() != 0){
                  int index = -1;
                  index = A[j].groupList.indexOf(new Integer(A[i].getGroup()));
                  if ( index != -1){
                     A[j].groupList.removeElementAt(index);
                     A[j].minusGroupNo();
                     if (A[j].getGroup() == A[i].getGroup() && A[j].getGroupNo() > 0){
                        A[j].setGroup(((Integer)(A[j].groupList.elementAt(0))).intValue());
                     }
                  }
               }
            }
         }
      }
   }

   static long calculateCost(){
      int moves = 0; /* total number of move operations */
      int no_of_nonempty_group = 0; /* number of total noneempty groups */
      long totalCost = 0; /* total cost of MoJo */

      /* find none empty groups and find total number of moves */
      for (int i = 0; i<l; i++){
         /* caculate the count of nonempty groups */
         /* when we found that a group was set to empty but in fact is not empty,
         we increase the number of noneempty group by 1 */
         if ( groupscount[A[i].getGroup()] == 0){
            no_of_nonempty_group += 1;
         }
         /* assign group tags */
         /* if this group has no tag, then we assign A[i] to its tag */
         if (grouptags[A[i].getGroup()] == null){
            grouptags[A[i].getGroup()] = A[i];
         }
         /* assign the group count */
         groupscount[A[i].getGroup()] += 1;
         /* calculate the number of move opts for each cluster */
         moves += A[i].gettotalTags() - A[i].getMaxtag();
      }
      totalCost = moves+l-no_of_nonempty_group;

      /* output the total cost */
      if (action == mojoPlus)
        System.out.print("MoJoPlus("+sourceFile+","+targetFile+") = ");
      else
        System.out.print("MoJo("+sourceFile+","+targetFile+") = ");

      System.out.println(totalCost);
      return totalCost;
   }

   static void showerrormsg(){
     System.out.println("");
     System.out.println("Please use correct syntax:");
     System.out.println("");
     System.out.println("java MoJo [-a -s -as -v - m -e] a.rsf b.rsf [r.rsf]");
     System.out.println("java MoJo a.rsf b.rsf");
     System.out.println("  calculates the MoJo distance between a.rsf and b.rsf");
     System.out.println("java MoJo -a a.rsf b.rsf");
     System.out.println("  calculates the MoJoPlus distance between a.rsf and b.rsf");
     System.out.println("java MoJo -s a.rsf b.rsf");
     System.out.println("  calculates the one-way MoJo distance from a.rsf to b.rsf");
     System.out.println("java MoJo -as a.rsf b.rsf");
     System.out.println("  calculates the one-way MoJoPlus distance from a.rsf to b.rsf");
     System.out.println("java MoJo -v a.rsf b.rsf");
     System.out.println("  outputs all the Move and Join operations to transform a.rsf to b.rsf");
     System.out.println("java MoJo -m a.rsf b.rsf");
     System.out.println("  calculates the MoJoFM metric between a.rsf and b.rsf");
     System.out.println("java MoJo -e a.rsf b.rsf r.rsf");
     System.out.println("  calculates the EdgeMoJo metric between a.rsf and b.rsf");
     System.exit(0);

   }
   /*tag assigment */
   static void tagAssigment(){
      for (int i = 0; i < l; i++)
      {
         int g = -1;
         String nameB = "";
         for (int j = 0; j < ((Vector)tempA.elementAt(i)).size(); j ++ ){
            String obj = (String)((Vector)tempA.elementAt(i)).elementAt(j);
            nameB = (String)tableObj.get(obj);
            if (nameB == null) {
              System.out.println("Error: object "+obj+"can not be found in target partition" );
              System.exit(0);
            }
            g = ((Integer)tableB.get(nameB)).intValue();
            if (action == mojoPlus)
              A[i].addobject_mojoplus(g,obj);
            else
              A[i].addobject(g,obj);
         }
      }

   }
   static void readSourceBunchFile(){
     try
     {
       br_s = new BufferedReader ( new FileReader(sourceFile) );
     }
     catch ( FileNotFoundException e)
     {
       System.out.println("source file "+sourceFile+" not found");
       System.exit(0);
     };
     try
     {
       for (String str_s = br_s.readLine(); str_s != null; str_s = br_s.readLine())
       {
         int equalMark = str_s.indexOf("=");
         String strClusterA = str_s.substring(0,equalMark).trim();
         String objList = str_s.substring(equalMark+1).trim();
         StringTokenizer st = new StringTokenizer(objList,",");
         int objNumber = st.countTokens();
         o += objNumber;
         int index = tableA.size();
         nameA.addElement(strClusterA);
         tableA.put(strClusterA,new Integer(index));
         tempA.addElement(new Vector());
         for (int i = 0; i < objNumber; i++)
         {
           String obj = st.nextToken().trim();
           ((Vector)tempA.elementAt(index)).addElement(obj);
         }
       }
     }
     catch (IOException e)
     {
       System.out.println("Error reading data file");
       System.exit(1);
     };
     try {
       br_s.close();
     }
     catch ( IOException e)
     {
       System.out.println(e.getMessage());
       System.exit(0);
     };
   }

     /* read the source file, put all the objects into vector tempA,tempA.elementOf(i)
     will be the cluster A[i].When we want to determine a objects tag, just use tableObj.get
     to determine which cluster in B the current objects belongs in */
   static void readSourceRSFfile(){
     try
     {
       br_s = new BufferedReader ( new FileReader(sourceFile) );
     }
     catch ( FileNotFoundException e)
     {
       System.out.println("source file "+sourceFile+" not found");
       System.exit(0);
     };
     try
     {
       for (String str_s = br_s.readLine(); str_s != null; str_s = br_s.readLine())
       {

         StringTokenizer st = new StringTokenizer(str_s);
         if ( st.countTokens() != 3) {
           System.out.println("wrong format in one line: doesn't contain 3 field");
           System.exit(0);
         };

         if ( !st.nextToken().toLowerCase().equals("contain")) {
           System.out.println("wrong format in one line: the first field must be contain");
           System.exit(0);
         };

         int index = -1;
         o++;
         String strClusterA = st.nextToken();
         String obj = st.nextToken();

         Object objectIndex = tableA.get(strClusterA);
         if ( objectIndex == null ) {
           index = tableA.size();
           nameA.addElement(strClusterA);
           tableA.put(strClusterA,new Integer(index));
           tempA.addElement(new Vector());
         }
         else
         {
           index = ((Integer)objectIndex).intValue();
         }
         ((Vector)tempA.elementAt(index)).addElement(obj);

       };
     }
     catch (IOException e)
     {
       System.out.println("Error reading data file");
       System.exit(1);
     };
     try {
       br_s.close();
     }
     catch ( IOException e)
     {
       System.out.println(e.getMessage());
       System.exit(0);
     };

   }
   static void readRelationRSFfile(){
     try
     {
       br_r = new BufferedReader ( new FileReader(relationFile) );
     }
     catch ( FileNotFoundException e)
     {
       System.out.println("relation file "+targetFile+" not found");
       System.exit(0);
     };
     try{
       for (String str_r = br_r.readLine(); str_r != null; str_r = br_r.readLine())
       {
         StringTokenizer st = new StringTokenizer(str_r);
         /* each line must contain 3 field, like call obj1 obj2 */
         if ( st.countTokens() != 3) {
           System.out.println("wrong format in one line: doesn't contain 3 field");
           System.exit(0);
         };
         /* currently, we accept all kinds of relation, so in fact the first token is useless */
         st.nextToken();
         /*
         if ( !st.nextToken().toLowerCase().equals("call")) {
           System.out.println("wrong format in one line: the first field must be call");
           System.exit(0);
         };*/

         String obj1 = st.nextToken();
         String obj2 = st.nextToken();
         /* we use obj1+"%@$"+obj2 as the key, store it into hash table, for all kinds of relationship
         we consider them with same connection strength */
         /* for example, if this time we see a call obj1 obj2, we store obj1+"%@$"+obj2 with value 1,
         next time we see a ref obj1 obj2, we store obj1+"%@$"+obj2 with value 2*/
         if ( tableR.get(obj1+"%@$"+obj2) == null)
           tableR.put(obj1+"%@$"+obj2, new Double(1));
         else
         {
           double previous_value = ((Double)(tableR.get(obj1+"%@$"+obj2))).doubleValue();
           tableR.put(obj1+"%@$"+obj2, new Double(previous_value+1));
         }

       }
     }
     catch (IOException e)
     {
       System.out.println("Error reading data file");
       System.exit(1);
     };

     try
     {
       br_r.close();
     }
     catch ( IOException e)
     {
       System.out.println(e.getMessage());
       System.exit(0);
     };

   }
   /*read target file */
   static void readTargetBunchFile(){
      /* read the target file, put all the objects into hash table tableObj,
      then we can get a object's tag easily when we read the source file, we also
      count the size of cluster B using hashtalbe tableB */
      try
      {
        br_t = new BufferedReader ( new FileReader(targetFile) );
      }
      catch ( FileNotFoundException e)
      {
        System.out.println("target file "+targetFile+" not found");
        System.exit(1);
      };

      try{
        for (String str_t = br_t.readLine(); str_t != null; str_t = br_t.readLine())
        {
          int equalMark = str_t.indexOf("=");
          String strClusterB = str_t.substring(0,equalMark).trim();
          String objList = str_t.substring(equalMark+1,str_t.length()).trim();
          StringTokenizer st = new StringTokenizer(objList,",");
          int objNumber = st.countTokens();

          /* this cluster is still not in the tableB, then we add it first */
          int index = tableB.size();
          nameB.addElement(strClusterB);
          /* the first time, it contain only one object, i.e., the current object */
          numberB.addElement(new Integer(objNumber));
          tableB.put(strClusterB,new Integer(index));

          for (int i = 0; i < objNumber; i++)
          {
            String obj = st.nextToken().trim();
            tableObj.put(obj, strClusterB);
          }
        }
      }
      catch (IOException e)
      {
        System.out.println("Error reading data file");
        System.exit(1);
      };
      try {
        br_t.close();
      }
      catch ( IOException e)
      {
        System.out.println(e.getMessage());
        System.exit(1);
      };

   }

   static void readTargetRSFFile(){
      /* read the target file, put all the objects into hash table tableObj,
      then we can get a object's tag easily when we read the source file, we also
      count the size of cluster B using hashtalbe tableB */
      try
      {
        br_t = new BufferedReader ( new FileReader(targetFile) );
      }
      catch ( FileNotFoundException e)
      {
        System.out.println("target file "+targetFile+" not found");
        System.exit(1);
      };

      try{
        for (String str_t = br_t.readLine(); str_t != null; str_t = br_t.readLine())
        {

          StringTokenizer st = new StringTokenizer(str_t);
          if ( st.countTokens() != 3) {
            System.out.println("wrong format in one line: doesn't contain 3 field");
            System.exit(0);
          };

          if ( !st.nextToken().toLowerCase().equals("contain")) {
            System.out.println("wrong format in one line: the first field must be contain");
            System.exit(0);
          };


          String strClusterB = st.nextToken();
          /* transfer "A_4" to A_4 */
          int first_quote_index = strClusterB.indexOf("\"");
          if ( first_quote_index == 0 && strClusterB.indexOf("\"",first_quote_index+1) == strClusterB.length()-1)
            strClusterB = strClusterB.substring(first_quote_index+1,strClusterB.length()-1);

          String obj = st.nextToken();
          int index = -1;


          /* try to find the cluster of the current object in the table */
          Object objectIndex = tableB.get(strClusterB);

          if ( objectIndex == null ) {
            /* this cluster is still not in the tableB, then we add it first */
            index = tableB.size();
            nameB.addElement(strClusterB);
            /* the first time, it contain only one object, i.e., the current object */
            numberB.addElement(new Integer(1));
            tableB.put(strClusterB,new Integer(index));

          }
          else
          {
            index = ((Integer)objectIndex).intValue();
            /* how many objects there are in the current cluster B_index, and we add one then store back*/
            int cur_number = 1 + ((Integer)numberB.elementAt(index)).intValue();
            /* set it back */
            numberB.setElementAt(new Integer(cur_number),index);


          }

          tableObj.put(obj, strClusterB);
        };
      }
      catch (IOException e)
      {
        System.out.println("Error reading data file");
        System.exit(1);
      };
      try {
        br_t.close();
      }
      catch ( IOException e)
      {
        System.out.println(e.getMessage());
        System.exit(1);
      };


   }
   static boolean isBunch(String file){
     int dot = file.lastIndexOf(".");
     if (dot < 0 ) return false;
     String extend = file.substring(dot+1).trim();
     if (extend.equalsIgnoreCase("bunch")) return true;
     return false;
   }
}

