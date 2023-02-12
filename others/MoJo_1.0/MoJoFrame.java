import java.awt.*;
import java.lang.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import java.io.*;
import java.util.*;

/**
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2002</p>
 * <p>Company: </p>
 * @author Zhihua
 * @version 1.0
 */

public class MoJoFrame extends JFrame {
  /* we show the process of MoJo */
  private final int singleDirection = 1;
  /* we calculate the MoJo metric */
  private final int doubleDirection = 2;

  /* we use mojoPlus*/
  private final int mojoPlus = 4;
  /*we use edge MoJo */
  private final int edgeMoJo = 8;
  /* none special action */
  private final int normalMoJo = 16;

  /* we show the process of MoJo */
  private final int showStatus = 32;
  /* we calculate the MoJo metric */
  private final int mojoMetric = 64;
  /*we separate omnipresent node for mojo calculation */
  private final int omniPresent = 128;

  /* the initial action */
  int action = normalMoJo;
  /*if we want to caculate both the distance between both A and B, then the */
  /* loopNum should be 2, otherwise should be 1 */
  int loopNum = doubleDirection;
  /* the running option */
  int option = 0;

  /* vector of object list */
  Vector objectVector;
  /* vector of omnipresent nodes*/
  Vector omniVector;
  /* number of connections for each object */
  Hashtable objectConnectionTable;
  /*number of connected clusters for each object */
  Hashtable objectConnectedClusterTable;
  /* average connections of each object */
  double averageConnections = 0;
  double averageConnectedClusters = 0;

  /* this hashtable is used to store the relation between object and cluster in B */
  Hashtable tableObj = new Hashtable();
  Hashtable tableAuthorativePatition_Obj = new Hashtable();
  /* countA & countB is used to calculate the different number of clusters in A & B */
  Hashtable tableA = new Hashtable();
  Hashtable tableB = new Hashtable();
  /*relation between objects */
  Hashtable tableR = new Hashtable();
  /*file name */
  String sourceFile = "",targetFile = "", relationFile = "";
  /* use for store the name of each items */
  Vector nameA = new Vector();
  Vector nameB = new Vector();
  /* use for store the number of ojbects in each cluster in partition B
     this is use in calculate the maxdistance from partition B */
  Vector numberB = new Vector();
  /* This vector is used to store all the objects and cluster in A */
  Vector tempA  = new Vector();
  Vector tempB  = new Vector();


  int l = 0;    /* number of clusters in A */
  int m = 0;    /* number of clusters in B */
  long o = 0;   /* number of total objects */
  long totalCost = 0; /*total cost of Mojo */
  cluster A[] = null;    /* A */

  /* record the capicity of each group, if the group is empty
  ,the count is zero, otherwise >= 1 */
  int groupscount[] = null;

  /* after join operations, each group will have only one cluster left,
  we use grouptags[i] to indicate the remain cluster in group i*/
  cluster grouptags[] = null; /* every none empty group have a tag point to a cluser in A */



  /* init the files */

  BufferedReader br_t; /* used for reading from target file */
  BufferedReader br_s; /* used for reading from traget file */
  BufferedReader br_r; /* used for reading from relation file */

  final int fileNameColumns = 10;
  private JFileChooser chooser = new JFileChooser(System.getProperty("user.dir"));
  private JPanel contentPane;
  private JMenuBar jMenuBar1 = new JMenuBar();
  private JMenu jMenuFile = new JMenu();
  private JMenuItem jMenuFileExit = new JMenuItem();
  private JMenu jMenuHelp = new JMenu();
  private JMenuItem jMenuHelpAbout = new JMenuItem();
  private JTabbedPane jTabbedPane = new JTabbedPane();
  private JPanel basicPanel = new JPanel();
  private JPanel optionPanel = new JPanel();
  private JPanel omnipresentPanel = new JPanel();
  private JTextField sourcejTextField = new JTextField(fileNameColumns);

  private Border border1;
  private Border border3;
  private Border border4;
  private Border border5;
  private Border border6;
  private Border border7;

  /* all the labels */
  private JLabel jLabel1 = new JLabel("  SourceFileName");
  private JLabel jLabel2 = new JLabel("  targetFileName");
  private JLabel jLabel3 = new JLabel("relationFileName");
  private JLabel jLabel4 = new JLabel();
  private JLabel jLabel5 = new JLabel("Direction of MoJo:");
  private JLabel jLabel6 = new JLabel("Type of MoJo:");
  private JLabel jLabel7 = new JLabel("Running Opition of MoJo:");
  private JLabel jLabel8 = new JLabel("0 Objects");
  private JLabel jLabel9 = new JLabel("0 Omnipresent Nodes");
  private JLabel jLabel10 = new JLabel("omnipresent objects with ");
  private JLabel jLabel11 = new JLabel("times more than the average connection");
  private JLabel jLabel13 = new JLabel("omnipresent objects with more than");
  private JLabel jLabel14 = new JLabel("connections");
  private JLabel jLabel15 = new JLabel("The following options can only be selected after you set the target partition file");
  private JLabel jLabel16 = new JLabel("omnipresent objects with more than");
  private JLabel jLabel17 = new JLabel("different connected clusters");
  private JLabel jLabel18 = new JLabel("omnipresent objects with ");
  private JLabel jLabel19 = new JLabel("times more than the average connected clusters");

  private JLabel jLabel12 = new JLabel();

  private JButton SourcejButton = new JButton(" ChooseSourceFile");
  private JTextField targetjTextField = new JTextField(fileNameColumns);
  private JButton targetjButton = new JButton(" ChooseTargetFile ");
  private JTextField relationjTextField = new JTextField(fileNameColumns);
  private JButton relationjButton = new JButton("ChooseRelationFile");
  private JButton runBtn = new JButton("Run");
  private JButton showaniBtn = new JButton("Show the animation");
  private JScrollPane jScrollPane1 = new JScrollPane();
  private JTextArea outTextArea = new JTextArea();

  private ButtonGroup omniSelectBtnGroup = new ButtonGroup();
  private ButtonGroup directionBtnGroup = new ButtonGroup();
  private JPanel directionPanel = new JPanel();
  private JRadioButton signleRadioBtn = new JRadioButton("Single direction MoJo");
  private JRadioButton doubleRadioBtn = new JRadioButton("Double direction MoJo");
  private JPanel mojoTypePanel = new JPanel();
  private ButtonGroup typeBtnGroup = new ButtonGroup();
  private JRadioButton EdgeRadioBtn = new JRadioButton("Edge MoJo");
  private JRadioButton MoJoPlusRadioBtn = new JRadioButton("MoJo Plus");
  private JRadioButton NormalRadioBtn = new JRadioButton("Normal MoJo");
  private JPanel runningoptionPanel = new JPanel();
  private JCheckBox showstatusCB = new JCheckBox("Show Running Status");
  private JCheckBox metricCB = new JCheckBox("Show MoJo Metric");
  private JCheckBox omniCB = new JCheckBox("Separate Omnipresent Nodes");

  private GridBagLayout gridBagLayout1 = new GridBagLayout();
  private GridBagLayout gridBagLayout2 = new GridBagLayout();
  private GridBagLayout gridBagLayout3 = new GridBagLayout();
  private GridBagLayout gridBagLayout4 = new GridBagLayout();
  private JPanel objPanel = new JPanel();
  private JPanel omniPanel = new JPanel();
  private JScrollPane objectScrollPane = new JScrollPane();
  private JList objectList = new JList();
  private JScrollPane omniScrollPane = new JScrollPane();
  private JList omniList = new JList();
  private GridBagLayout gridBagLayout5 = new GridBagLayout();
  private GridBagLayout gridBagLayout6 = new GridBagLayout();
  private GridBagLayout gridBagLayout7 = new GridBagLayout();
  private JPanel btnPanel = new JPanel();
  private JButton addBtn = new JButton("->");
  private JButton delBtn = new JButton("<-");
  private GridBagLayout gridBagLayout8 = new GridBagLayout();
  private JPanel omniOptionPanel = new JPanel();
  private JRadioButton relConRB = new JRadioButton("Relative connections");
  private JRadioButton ablConRB = new JRadioButton("Absolute connections");
  private JRadioButton realClusterConRB = new JRadioButton("Absolute connected clusters");
  private JRadioButton avgClusterConRB = new JRadioButton("Relative connected clusters");


  private JButton searchBtn = new JButton("Search");
  private JButton setAuthorativeBtn = new JButton("Set the authoritative Partition ");
  private JTextField timesTextField = new JTextField();
  private JTextField ablTextField = new JTextField();
  private JTextField realClusterTextField = new JTextField();
  private JTextField avgClusterTextField = new JTextField();

  //Construct the frame
  public MoJoFrame() {
    enableEvents(AWTEvent.WINDOW_EVENT_MASK);
    try {
      jbInit();
    }
    catch(Exception e) {
      e.printStackTrace();
    }
  }
  //Component initialization
  private void jbInit() throws Exception  {
    //setIconImage(Toolkit.getDefaultToolkit().createImage(MoJoFrame.class.getResource("[Your Icon]")));
    contentPane = (JPanel) this.getContentPane();

    border1 = BorderFactory.createCompoundBorder(new EtchedBorder(EtchedBorder.RAISED,new Color(218, 218, 218),new Color(107, 107, 107)),BorderFactory.createEmptyBorder(10,10,10,10));
    border3 = BorderFactory.createCompoundBorder(new EtchedBorder(EtchedBorder.RAISED,Color.white,new Color(178, 178, 178)),BorderFactory.createEmptyBorder(5,5,5,5));
    border4 = BorderFactory.createCompoundBorder(new EtchedBorder(EtchedBorder.RAISED,Color.white,new Color(142, 142, 142)),BorderFactory.createEmptyBorder(30,30,30,30));
    border5 = BorderFactory.createCompoundBorder(new EtchedBorder(EtchedBorder.RAISED,Color.white,new Color(142, 142, 142)),BorderFactory.createEmptyBorder(30,30,30,30));
    border6 = BorderFactory.createCompoundBorder(new EtchedBorder(EtchedBorder.RAISED,Color.white,new Color(142, 142, 142)),BorderFactory.createEmptyBorder(50,50,50,50));
    border7 = BorderFactory.createCompoundBorder(BorderFactory.createBevelBorder(BevelBorder.RAISED,Color.white,Color.white,new Color(124, 124, 124),new Color(178, 178, 178)),BorderFactory.createEmptyBorder(2,2,2,2));

    doubleRadioBtn.setSelected(true);

    directionPanel.setLayout(gridBagLayout1);
    mojoTypePanel.setLayout(gridBagLayout2);
    runningoptionPanel.setLayout(gridBagLayout3);
    optionPanel.setLayout(gridBagLayout4);
    jLabel8.setToolTipText("");
    jLabel8.setDisplayedMnemonic('0');
    objectList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
    objectList.setToolTipText("");
    objectList.setFixedCellWidth(40);
    objPanel.setLayout(gridBagLayout5);
    omniPanel.setLayout(gridBagLayout6);
    omniList.setToolTipText("");
    omniList.setFixedCellWidth(40);
    objectScrollPane.setPreferredSize(new Dimension(300, 300));
    omniScrollPane.setPreferredSize(new Dimension(300, 300));
    omnipresentPanel.setLayout(gridBagLayout7);
    delBtn.setToolTipText("");

    addBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        addBtn_actionPerformed(e);
      }
    });
    doubleRadioBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        doubleRadioBtn_actionPerformed(e);
      }
    });
    signleRadioBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        signleRadioBtn_actionPerformed(e);
      }
    });
    delBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        delBtn_actionPerformed(e);
      }
    });
    btnPanel.setLayout(gridBagLayout8);
    NormalRadioBtn.setSelected(true);
    NormalRadioBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        NormalRadioBtn_actionPerformed(e);
      }
    });
    searchBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        searchBtn_actionPerformed(e);
      }
    });
    setAuthorativeBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        setAuthorativeBtn_actionPerformed(e);
      }
    });


    timesTextField.setBorder(border7);
    ablTextField.setBorder(border7);
    realClusterTextField.setBorder(border7);
    avgClusterTextField.setBorder(border7);
    timesTextField.setText("3.0");
    ablTextField.setText("5");
    realClusterTextField.setText("5");
    avgClusterTextField.setText("3");
    avgClusterTextField.setColumns(5);
    timesTextField.setColumns(5);
    ablTextField.setColumns(5);
    realClusterTextField.setColumns(5);

    EdgeRadioBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        EdgeRadioBtn_actionPerformed(e);
      }
    });
    MoJoPlusRadioBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        MoJoPlusRadioBtn_actionPerformed(e);
      }
    });
    metricCB.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
      }
    });
    omniCB.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
      }
    });
    showstatusCB.addItemListener(new java.awt.event.ItemListener() {
      public void itemStateChanged(ItemEvent e) {
        showstatusCB_itemStateChanged(e);
      }
    });
    omniCB.addItemListener(new java.awt.event.ItemListener() {
      public void itemStateChanged(ItemEvent e) {
        omniCB_itemStateChanged(e);
      }
    });
    metricCB.addItemListener(new java.awt.event.ItemListener() {
      public void itemStateChanged(ItemEvent e) {
        metricCB_itemStateChanged(e);
      }
    });
    runBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        runBtn_actionPerformed(e);
      }
    });
    showaniBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        showaniBtn_actionPerformed(e);
      }
    });


    outTextArea.setEditable(false);
    outTextArea.setRows(20);
    omniCB.setEnabled(false);
    directionBtnGroup.add(signleRadioBtn);
    directionBtnGroup.add(doubleRadioBtn);


    this.setSize(new Dimension(800, 600));
    this.setResizable(false);
    this.setTitle("MoJo");
    jMenuFile.setText("File");
    jMenuFileExit.setText("Exit");
    jMenuFileExit.addActionListener(new ActionListener()  {
      public void actionPerformed(ActionEvent e) {
        jMenuFileExit_actionPerformed(e);
      }
    });
    jMenuHelp.setText("Help");
    jMenuHelpAbout.setText("About");
    jMenuHelpAbout.addActionListener(new ActionListener()  {
      public void actionPerformed(ActionEvent e) {
        jMenuHelpAbout_actionPerformed(e);
      }
    });
    jTabbedPane.setBorder(border1);
    jTabbedPane.setToolTipText("");
    SourcejButton.setToolTipText("");
    SourcejButton.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        SourcejButton_actionPerformed(e);
      }
    });
    //sourcejTextField.setEditable(false);
    //sourcejTextField.setText("         ");
    //targetjTextField.setEditable(false);
    //targetjTextField.setText("  ");
    jLabel2.setToolTipText("");
    targetjButton.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        targetjButton_actionPerformed(e);
      }
    });
    relationjTextField.setEditable(false);
    relationjTextField.setText("        ");
    relationjButton.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(ActionEvent e) {
        relationjButton_actionPerformed(e);
      }
    });
    contentPane.setEnabled(true);
    outTextArea.setBorder(border3);
    outTextArea.setColumns(20);
    directionPanel.setBorder(border4);
    typeBtnGroup.add(NormalRadioBtn);
    typeBtnGroup.add(EdgeRadioBtn);
    typeBtnGroup.add(MoJoPlusRadioBtn);
    mojoTypePanel.setBorder(border5);
    runningoptionPanel.setBorder(border6);
    jMenuFile.add(jMenuFileExit);
    jMenuHelp.add(jMenuHelpAbout);
    jMenuBar1.add(jMenuFile);
    jMenuBar1.add(jMenuHelp);
    contentPane.add(jTabbedPane);
    jTabbedPane.add(basicPanel, "Basic");
    jTabbedPane.add(optionPanel, "Option");
    jTabbedPane.add(omnipresentPanel,  "Omnipresent Node");
    jTabbedPane.setEnabledAt(jTabbedPane.indexOfComponent(omnipresentPanel),false);

    //set layout
    GridBagLayout gridBag = new GridBagLayout();
    basicPanel.setLayout(gridBag);
    GridBagConstraints c = new GridBagConstraints();

    c.fill = GridBagConstraints.HORIZONTAL;
    c.weightx = 1.0;
    c.weighty = 0.0;


    //basic panel
    c.gridwidth = 2;
    c.weightx = 2.0;
    basicPanel.add(sourcejTextField,c);
    c.weightx = 0.0;
    c.gridwidth = 1;
    basicPanel.add(jLabel1,c);
    c.gridwidth = GridBagConstraints.REMAINDER;
    basicPanel.add(SourcejButton,c);

    c.gridwidth = 2;
    c.weightx = 2.0;
    basicPanel.add(targetjTextField,c);
    c.weightx = 0.0;
    c.gridwidth = 1;
    basicPanel.add(jLabel2,c);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    basicPanel.add(targetjButton,c);

    c.gridwidth = 2;
    c.weightx = 2.0;
    basicPanel.add(relationjTextField,c);
    c.weightx = 0.0;
    c.gridwidth = 1;
    basicPanel.add(jLabel3,c);
    c.gridwidth = GridBagConstraints.REMAINDER;
    basicPanel.add(relationjButton,c);

    c.gridwidth = 1;
    c.weightx = 0.0;
    basicPanel.add(runBtn,c);
    basicPanel.add(showaniBtn,c);
    c.gridwidth = GridBagConstraints.REMAINDER;
    c.weightx = 1;
    basicPanel.add(jLabel4,c);
    basicPanel.add(jScrollPane1, c);
    jScrollPane1.getViewport().add(outTextArea, null);

    //option tab
    optionPanel.add(directionPanel,c);
    optionPanel.add(mojoTypePanel,c);
    optionPanel.add(runningoptionPanel,c);

    directionPanel.add(jLabel5,c);
    directionPanel.add(signleRadioBtn,c);
    directionPanel.add(doubleRadioBtn,c);

    mojoTypePanel.add(jLabel6,c);
    mojoTypePanel.add(NormalRadioBtn,c);
    mojoTypePanel.add(EdgeRadioBtn,c);
    mojoTypePanel.add(MoJoPlusRadioBtn,c);
    runningoptionPanel.add(jLabel7,c);
    runningoptionPanel.add(showstatusCB,c);
    runningoptionPanel.add(metricCB,c);
    runningoptionPanel.add(omniCB,c);

    //omnipresent tab
    omniPanel.add(jLabel9,c);
    omniPanel.add(omniScrollPane,c);
    objPanel.add(jLabel8,c);
    objPanel.add(objectScrollPane,c);
    omniScrollPane.getViewport().add(omniList, null);
    objectScrollPane.getViewport().add(objectList, null);

    c.gridwidth = 2;
    c.weightx = 1;
    omnipresentPanel.add(objPanel,c);
    c.weightx = 0;
    c.gridwidth = 1;
    omnipresentPanel.add(btnPanel,c);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    btnPanel.add(delBtn,c);
    btnPanel.add(addBtn,c);
    c.weightx = 1;
    omnipresentPanel.add(omniPanel,c);
    c.gridwidth = 1;
    c.weightx = 0;
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    omnipresentPanel.add(omniOptionPanel,c);
    omniOptionPanel.setLayout(new GridBagLayout());
    c.gridwidth = 1;
    omniOptionPanel.add(searchBtn, c);
    omniOptionPanel.add(setAuthorativeBtn,c);
    c.weightx = 1.0;
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    omniOptionPanel.add(jLabel12, c);

    c.gridwidth = 1;
    c.weightx = 0;
    omniOptionPanel.add(relConRB, c);
    omniOptionPanel.add(jLabel10, c);
    omniOptionPanel.add(timesTextField, c);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    omniOptionPanel.add(jLabel11, c);

    c.gridwidth = 1;
    c.weightx = 0;
    omniOptionPanel.add(ablConRB, c);
    omniOptionPanel.add(jLabel13, c);
    omniOptionPanel.add(ablTextField, c);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    omniOptionPanel.add(jLabel14, c);
    omniOptionPanel.add(jLabel15, c);

    c.gridwidth = 1;
    c.weightx = 0;
    omniOptionPanel.add(realClusterConRB, c);
    omniOptionPanel.add(jLabel16, c);
    omniOptionPanel.add(realClusterTextField, c);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    omniOptionPanel.add(jLabel17, c);

    c.gridwidth = 1;
    c.weightx = 0;
    omniOptionPanel.add(avgClusterConRB, c);
    omniOptionPanel.add(jLabel18, c);
    omniOptionPanel.add(avgClusterTextField, c);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    omniOptionPanel.add(jLabel19, c);

    omniSelectBtnGroup.add(ablConRB);
    omniSelectBtnGroup.add(relConRB);
    omniSelectBtnGroup.add(realClusterConRB);
    omniSelectBtnGroup.add(avgClusterConRB);
    relConRB.setSelected(true);

    this.setJMenuBar(jMenuBar1);

  }
  //File | Exit action performed
  public void jMenuFileExit_actionPerformed(ActionEvent e) {
    System.exit(0);
  }
  //Help | About action performed
  public void jMenuHelpAbout_actionPerformed(ActionEvent e) {
    MoJoFrame_AboutBox dlg = new MoJoFrame_AboutBox(this);
    Dimension dlgSize = dlg.getPreferredSize();
    Dimension frmSize = getSize();
    Point loc = getLocation();
    dlg.setLocation((frmSize.width - dlgSize.width) / 2 + loc.x, (frmSize.height - dlgSize.height) / 2 + loc.y);
    dlg.setModal(true);
    dlg.pack();
    dlg.show();
  }
  //Overridden so we can exit when window is closed
  protected void processWindowEvent(WindowEvent e) {
    super.processWindowEvent(e);
    if (e.getID() == WindowEvent.WINDOW_CLOSING) {
      jMenuFileExit_actionPerformed(null);
    }
  }

  /* selecting the files */
  void SourcejButton_actionPerformed(ActionEvent e) {

    int retval = chooser.showDialog(this, SourcejButton.getText());
    if (retval == JFileChooser.APPROVE_OPTION)
    {
      if ( chooser.getSelectedFile().exists()){
        sourcejTextField.setText(chooser.getSelectedFile().getAbsolutePath());
        sourceFile = sourcejTextField.getText();
      }
      else
      JOptionPane.showMessageDialog(this,"File doesn't exists");
    }
  }

  void targetjButton_actionPerformed(ActionEvent e) {
    int retval = chooser.showDialog(this, targetjButton.getText());
    if (retval == JFileChooser.APPROVE_OPTION)
    {
      if ( chooser.getSelectedFile().exists()){
        targetjTextField.setText(chooser.getSelectedFile().getAbsolutePath());
        targetFile = targetjTextField.getText();
      }
      else
        JOptionPane.showMessageDialog(this,"File doesn't exists");
    }
  }

  void relationjButton_actionPerformed(ActionEvent e) {
    /* init the relashion table */
    tableR = new Hashtable();
    /* vector of object list */
    Vector tempobjectVector = new Vector();
    Hashtable tempobjectConnectionTable = new Hashtable();
    /* total connections, equal to total lines in relationship file */
    int totalConnections = 0;

    int retval = chooser.showDialog(this, relationjButton.getText());
    if (retval == JFileChooser.APPROVE_OPTION)
    {
      if ( chooser.getSelectedFile().exists()){
        relationjTextField.setText(chooser.getSelectedFile().getAbsolutePath());
        relationFile = relationjTextField.getText();

        /* we try to open the relationship file first */
        try
        {
          br_r = new BufferedReader ( new FileReader(relationFile) );
        }
        catch ( FileNotFoundException ef)
        {
          JOptionPane.showMessageDialog(this,"relation file "+relationFile+" not found");
          relationjTextField.setText("");
          relationFile = "";
          return;
        };

        try{
          for (String str_r = br_r.readLine(); str_r != null; str_r = br_r.readLine())
          {
            totalConnections++;
            StringTokenizer st = new StringTokenizer(str_r);
            /* each line must contain 3 field, like call obj1 obj2 */
            if ( st.countTokens() != 3) {
              //fail
              JOptionPane.showMessageDialog(this,"wrong format in one line: doesn't contain 3 field");
              relationjTextField.setText("");
              relationFile = "";
              return;
            };
            /* currently, we accept all kinds of relation, so in fact the first token is useless */
            st.nextToken();
            /*
            if ( !st.nextToken().toLowerCase().equals("call")) {
              JOptionPane.showMessageDialog(this,"wrong format in one line: the first field must be call");
              return;
            };*/

            String obj1 = st.nextToken();
            String obj2 = st.nextToken();
            if (tempobjectVector.indexOf(obj1)  == -1) tempobjectVector.add(obj1);
            if (tempobjectVector.indexOf(obj2)  == -1) tempobjectVector.add(obj2);

            if (tempobjectConnectionTable.get(obj1) == null)
              tempobjectConnectionTable.put(obj1,new Double(1));
            else
            {
              double previous_value = ((Double)tempobjectConnectionTable.get(obj1)).doubleValue();
              tempobjectConnectionTable.put(obj1,new Double(previous_value+1));
            }

            if (tempobjectConnectionTable.get(obj2) == null)
              tempobjectConnectionTable.put(obj2,new Double(1));
            else
            {
              double previous_value = ((Double)tempobjectConnectionTable.get(obj2)).doubleValue();
              tempobjectConnectionTable.put(obj2,new Double(previous_value+1));
            }

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
        catch (IOException ef)
        {
          //fail
          JOptionPane.showMessageDialog(this,"Error reading data file");
          relationjTextField.setText("");
          relationFile = "";
          return;
        };

        try
        {
          br_r.close();
        }
        catch ( IOException ef)
        {
          JOptionPane.showMessageDialog(this,ef.getMessage());
          //fail
          relationjTextField.setText("");
          relationFile = "";
          return;
        };
        //sucess
        objectVector = tempobjectVector;
        averageConnections = (double)totalConnections / (double)objectVector.size();
        objectConnectionTable = tempobjectConnectionTable;
        objectList.setListData(objectVector);
        jLabel8.setText(""+ objectVector.size()+" Objects");
        omniVector = new Vector();
        omniList.setListData(omniVector);
        jTabbedPane.setEnabledAt(jTabbedPane.indexOfComponent(omnipresentPanel),true);
        realClusterConRB.setEnabled(false);
        avgClusterConRB.setEnabled(false);
        relConRB.setSelected(true);


      }
      else
        JOptionPane.showMessageDialog(this,"File doesn't exists");
    }
  }

  /* set the target authorative partition */
  void setAuthorativeBtn_actionPerformed(ActionEvent e) {
    int retval = chooser.showDialog(this, setAuthorativeBtn.getText());
    if (retval == JFileChooser.APPROVE_OPTION)
    {
      if ( chooser.getSelectedFile().exists()){

        String fileName = chooser.getSelectedFile().getAbsolutePath();
        if (isBunch(fileName)){
          if (!readAuthorativePatitionBunchFile(fileName)) return;
        }
        else{
          if (!readAuthorativePatitionRSFFile(fileName)) return;
        }
        jLabel12.setText(fileName);
        Hashtable tempobjectConnectedClusterTable = new Hashtable();
        Hashtable tempTable = new Hashtable();
        int totalConnectedClusters = 0;
        //temporary add all the elements to objectVector
        for ( int j = 0; j < omniVector.size(); j++)
          objectVector.add(omniVector.elementAt(j));

        for (int i = 0; i < objectVector.size(); i++){
          for ( int j = i + 1; j < objectVector.size(); j++){
            String obj1 = (String)objectVector.get(i);
            String obj2 = (String)objectVector.get(j);
            if ( tableR.get(obj1+"%@$"+obj2) != null || tableR.get(obj2+"%@$"+obj1) != null){
              String Cluster1 = (String)tableAuthorativePatition_Obj.get(obj1);
              String Cluster2 = (String)tableAuthorativePatition_Obj.get(obj2);
              if ( tempTable.get(obj1+"%@$"+Cluster2) == null){
              	totalConnectedClusters++;
                tempTable.put(obj1+"%@$"+Cluster2,"yes");
                if ( tempobjectConnectedClusterTable.get(obj1) == null){
                  tempobjectConnectedClusterTable.put(obj1,new Double(1));
                }
                else
                {
                  double previous_value = ((Double)tempobjectConnectedClusterTable.get(obj1)).doubleValue();
                  tempobjectConnectedClusterTable.put(obj1,new Double(previous_value+1));
                }

              };

              if ( tempTable.get(obj2+"%@$"+Cluster1)== null){
              	totalConnectedClusters++;
                tempTable.put(obj2+"%@$"+Cluster1,"yes");
                if ( tempobjectConnectedClusterTable.get(obj2) == null){
                  tempobjectConnectedClusterTable.put(obj2,new Double(1));
                }
                else
                {
                  double previous_value = ((Double)tempobjectConnectedClusterTable.get(obj2)).doubleValue();
                  tempobjectConnectedClusterTable.put(obj2,new Double(previous_value+1));
                }
              };

            }
          }
        }
        //delete all the object in omnipresent nodes from object vector
        for ( int j = 0; j < omniVector.size(); j++)
          objectVector.removeElementAt(objectVector.size()-1);


        objectList.setListData(objectVector);
        omniList.setListData(omniVector);
        realClusterConRB.setEnabled(true);
        avgClusterConRB.setEnabled(true);
        averageConnectedClusters = (double)totalConnectedClusters / (double)objectVector.size();
        objectConnectedClusterTable = tempobjectConnectedClusterTable;

      }
      else
      JOptionPane.showMessageDialog(this,"File doesn't exists");
    }

  }
  void searchBtn_actionPerformed(ActionEvent e) {
    if (objectVector == null) return;
    double times = Double.parseDouble(timesTextField.getText());
    double value = Double.parseDouble(ablTextField.getText());
    double clusters = Double.parseDouble(realClusterTextField.getText());
    double timesClusters = Double.parseDouble(avgClusterTextField.getText());
    int i = 0;

    if ( omniVector.size() > 0 ){
      int result = JOptionPane.showConfirmDialog(this, "Do you want to clear the current Omnipresent Node list?", "Omnipresent Nodes alreay exist", JOptionPane.YES_NO_CANCEL_OPTION, JOptionPane.INFORMATION_MESSAGE);
      if ( result == JOptionPane.CANCEL_OPTION ) return;
      if ( result == JOptionPane.YES_OPTION){
        for (i = 0; i < omniVector.size(); i ++){
          objectVector.add(omniVector.elementAt(i));
        };
        i = 0;
        omniVector.clear();
        jLabel9.setText(""+ omniVector.size()+" Omnipresent Nodes");
        jLabel8.setText(""+ objectVector.size()+" Objects");
      }
    };

    if (relConRB.isSelected()){
      while( i < objectVector.size()){
        if ( ((Double)objectConnectionTable.get(objectVector.get(i))).doubleValue() > times * averageConnections){
          omniVector.add(objectVector.get(i));
          objectVector.remove(i);
        }
        else i++;
      }
    };
    if (ablConRB.isSelected()){
      while( i < objectVector.size()){
        if ( ((Double)objectConnectionTable.get(objectVector.get(i))).doubleValue() > value){
          omniVector.add(objectVector.get(i));
          objectVector.remove(i);
        }
        else i++;
      }
    };

    if (avgClusterConRB.isSelected()){
      while( i < objectVector.size()){
        if ( ((Double)objectConnectedClusterTable.get(objectVector.get(i))).doubleValue() > timesClusters * averageConnectedClusters){
          omniVector.add(objectVector.get(i));
          objectVector.remove(i);
        }
        else i++;
      }
    };

    if (realClusterConRB.isSelected()){
      while( i < objectVector.size()){
        if ( ((Double)objectConnectedClusterTable.get(objectVector.get(i))).doubleValue() > clusters ){
          omniVector.add(objectVector.get(i));
          objectVector.remove(i);
        }
        else i++;
      }
    };
    jLabel9.setText(""+ omniVector.size()+" Omnipresent Nodes");
    jLabel8.setText(""+ objectVector.size()+" Objects");
    objectList.setListData(objectVector);
    omniList.setListData(omniVector);
  }


  String maxbipartiteMatching(cluster[] A,int l,int m,int action){
     String str = "";
     /* create the graph and add all the edges */
     graph bgraph = new graph(l+m,l,m);

     for ( int i = 0; i < l; i++){
        for (int j = 0; j < A[i].groupList.size(); j ++ ){
           bgraph.addedge(i,l+((Integer)A[i].groupList.elementAt(j)).intValue());
        }
     }

     /* use max bipature to caculate the group */
     str += bgraph.Matching(hasOption(showStatus));
     /* assign group after matching, for each Ai in matching, assign the corresponding group,
     for other cluster in A, just leave them alone */
     for ( int i = l; i < l + m; i ++){
        if ( bgraph.vertex[i].mathced ){
           int index = ((Integer)bgraph.adjacentList[i].elementAt(0)).intValue();
           A[index].setGroup(i-l);
           if (hasOption(showStatus)) str += "Assign A"+(index+1)+" to group G"+(i-l+1)+"\n";
        }
      }
      return str;
  }

   /* calculate the mojo metirc distance value, using the formula
   Q(M) = 1 - mno(A,B)/ max(mno(any_A,B)) * 100% */
   double mojoMetricValue(long maxDis, Vector number_of_B, long obj_number,long totalCost){
     return Math.rint(( 1 - (double)totalCost / (double)maxDis)*10000) / 100;
   }

   /* calculate the max(mno(B, any_A)), which is also the max(mno(any_A, B)) */
   long maxDistanceTo(Vector number_of_B, long obj_number) {
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
   /* optimization */
   /* this function is unnecessary. but it can accelerate the process of mojo */
   /* we believe that most cluster in A has only one unique group to select, so we
   can assign the group to them and kick them out. thus few clusters and edges was
   left in graph */

  void optimization(cluster[] A,int l){

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

  long calculateCost(int l,int[] groupscount,cluster[] A,cluster[] grouptags, String f1, String f2){
     String str = "";
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


     if (action == mojoPlus)
       str += "MoJoPlus("+f1+","+f2+") = ";
     else
       str += "MoJo("+f1+","+f2+") = ";

     str += totalCost + "\n";
     outTextArea.append(str);
     return totalCost;
  }

  /*tag assigment */
  boolean tagAssigment(int l,int m,Vector tempA,Hashtable tableObj,Hashtable tableB,cluster[] A){
    for (int i = 0; i < l; i++)
    {
      int g = -1;
      String nameB = "";
      for (int j = 0; j < ((Vector)tempA.elementAt(i)).size(); j ++ ){
        String obj = (String)((Vector)tempA.elementAt(i)).elementAt(j);
        nameB = (String)tableObj.get(obj);
        if (nameB == null) {
          JOptionPane.showMessageDialog(this,"Error: object "+obj+"can not be found in target partition" );
          return false;
        }

        g = ( (Integer) tableB.get(nameB)).intValue();
        if (action == mojoPlus)
          A[i].addobject_mojoplus(g,obj);
        else
          A[i].addobject(g,obj);
      }

    }
    return true;

  }
  /* try to locate whether we have this option */
  boolean hasOption(int value){
    int temp = option;
    int doublevalue = value * 2;
    temp = temp % doublevalue;
    if (temp < value) return false;
    else return true;

  }

  void signleRadioBtn_actionPerformed(ActionEvent e) {
    loopNum = singleDirection;
  }

  void doubleRadioBtn_actionPerformed(ActionEvent e) {
    loopNum = doubleDirection;
  }

/* 3 type of MoJo action */
  void NormalRadioBtn_actionPerformed(ActionEvent e) {
    action = normalMoJo;
    metricCB.setEnabled(true);
    showstatusCB.setEnabled(true);
    omniCB.setSelected(false);
    omniCB.setEnabled(false);
  }

  void EdgeRadioBtn_actionPerformed(ActionEvent e) {
    action = edgeMoJo;
    showstatusCB.setEnabled(false);
    showstatusCB.setSelected(false);
    metricCB.setEnabled(false);
    metricCB.setSelected(false);
    omniCB.setEnabled(true);
  }

  void MoJoPlusRadioBtn_actionPerformed(ActionEvent e) {
    action = mojoPlus;
    showstatusCB.setEnabled(false);
    showstatusCB.setSelected(false);
    metricCB.setEnabled(true);
    omniCB.setSelected(false);
    omniCB.setEnabled(false);
  }

/* option of running MoJo */

  void showstatusCB_itemStateChanged(ItemEvent e) {
    if (e.getStateChange() == ItemEvent.SELECTED)
    {
     if (!hasOption(showStatus)) option += showStatus;
     JOptionPane.showMessageDialog(this,"In practical use because we only want to know the minmium number of join + move,we dont need to know the exact process, which is also very slow for large file.");
    };
    if (e.getStateChange() == ItemEvent.DESELECTED)
    {
      if (hasOption(showStatus)) option -= showStatus;
      //JOptionPane.showMessageDialog(this,"fuck");
    }

  }

  void omniCB_itemStateChanged(ItemEvent e) {
    if (e.getStateChange() == ItemEvent.SELECTED)
    {
      if (!hasOption(omniPresent)) option += omniPresent;
    };
    if (e.getStateChange() == ItemEvent.DESELECTED)
    {
      if (hasOption(omniPresent)) option -= omniPresent;
    };
  }

/* select or unselect mojo metric measure option */
  void metricCB_itemStateChanged(ItemEvent e) {

    if (e.getStateChange() == ItemEvent.SELECTED)
    {
      if (!hasOption(mojoMetric)) option += mojoMetric;
    };
    if (e.getStateChange() == ItemEvent.DESELECTED)
    {
      if (hasOption(mojoMetric)) option -= mojoMetric;
    };
  }

/* add the object from object list to omnipresent list */
  void addBtn_actionPerformed(ActionEvent e) {
    int index = objectList.getSelectedIndex();
    if (index != -1){
      omniVector.add(objectVector.get(index));
      objectVector.remove(index);
      jLabel9.setText(""+ omniVector.size()+" Omnipresent Nodes");
      jLabel8.setText(""+ objectVector.size()+" Objects");

      objectList.setListData(objectVector);
      omniList.setListData(omniVector);
    }

  }

/* remove the object from omnipresent list back to object list */
  void delBtn_actionPerformed(ActionEvent e) {
    int index = omniList.getSelectedIndex();
    if (index != -1){
      objectVector.add(omniVector.get(index));
      jLabel8.setText(""+ objectVector.size()+" Objects");
      omniVector.remove(index);
      jLabel9.setText(""+ omniVector.size()+" Omnipresent Nodes");

      objectList.setListData(objectVector);
      omniList.setListData(omniVector);
    };

  }

  void showaniBtn_actionPerformed(ActionEvent e){
    /*init the variables*/
    Object[] possibleValues = { "from cluster in specified files", "from random created clusters" };
    Object selectedValue = JOptionPane.showInputDialog(null, "Choose what type clusters you want to see the animation", "Select the Animation Type", JOptionPane.INFORMATION_MESSAGE, null, possibleValues, possibleValues[0]);
    if ( selectedValue == "from random created clusters")
    {
      mojoAnimation frame = new mojoAnimation(this);
      //Validate frames that have preset sizes
      //Pack frames that have useful preferred size info, e.g. from their layout
      //  frame.pack();
      frame.validate();
      //Center the window
      Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
      Dimension frameSize = new Dimension(800,600);
      if (frameSize.height > screenSize.height) {
        frameSize.height = screenSize.height;
      }
      if (frameSize.width > screenSize.width) {
        frameSize.width = screenSize.width;
      }
      frame.setLocation((screenSize.width - frameSize.width) / 2, (screenSize.height - frameSize.height) / 2);
      frame.setSize(frameSize);
      frame.setVisible(true);
      frame.setResizable(false);
      this.setEnabled(false);

      try {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
      }
      catch(Exception ef) {
        ef.printStackTrace();
      }

    };

    if ( selectedValue == "from cluster in specified files" )
    {
      init_var();

    /* check the availiablity of source file, target file and relationfile */
      if (sourceFile.trim().equals("")){
        JOptionPane.showMessageDialog(this,"Please specify the source file.");
        return;
      };

      if (targetFile.trim().equals("")) {
        JOptionPane.showMessageDialog(this,"Please specify the target file.");
        return;
      };

    /*read the target file */
      if (isBunch(targetFile))
      {
        if (!readTargetBunchFile())return;
      }
      else
      {
        if (!readTargetRSFFile()) return;
      };
    /*read the source file */
      if (isBunch(sourceFile))
      {
        if (!readSourceBunchFile())return;
      }
      else
      {
        if (!readSourceRSFFile()) return;
      };

      l = tableA.size(); /* number of clusters in A */
      m = tableB.size(); /* number of clusters in B */
      if ( o != tableObj.size())
      {
        JOptionPane.showMessageDialog(this,"The total number of ojbects in source file does not equal the total number of objects in target file");
        return;
      }; /* number of total objects */
      if ( l > 5 || m > 5 || o > 50)
      {
        JOptionPane.showMessageDialog(this,"Sorry, too much clusters or objects, can't display. L <= 5 M <= 5 O <= 50");
        return;
      };



      mojoAnimation frame = new mojoAnimation(this,tempA,tempB,o);
      //Validate frames that have preset sizes
      //Pack frames that have useful preferred size info, e.g. from their layout
      //  frame.pack();
      frame.validate();
      //Center the window
      Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
      Dimension frameSize = new Dimension(800,600);
      if (frameSize.height > screenSize.height) {
        frameSize.height = screenSize.height;
      }
      if (frameSize.width > screenSize.width) {
        frameSize.width = screenSize.width;
      }
      frame.setLocation((screenSize.width - frameSize.width) / 2, (screenSize.height - frameSize.height) / 2);
      frame.setSize(frameSize);
      frame.setVisible(true);
      frame.setResizable(false);
      this.setEnabled(false);

      try {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
      }
      catch(Exception ef) {
        ef.printStackTrace();
      }
    }

  }
/*running the program */
  void runBtn_actionPerformed(ActionEvent e) {

    /*init the variables*/
    init_var();

    /* check the availiablity of source file, target file and relationfile */
    if (sourceFile.trim().equals("")){
      JOptionPane.showMessageDialog(this,"Please specify the source file.");
      return;
    };

    if (targetFile.trim().equals("")) {
      JOptionPane.showMessageDialog(this,"Please specify the target file.");
      return;
    };

    if (action == edgeMoJo && relationFile.trim().equals("")){
      JOptionPane.showMessageDialog(this,"Please specify the relationship file.");
      return;
    };

    for ( int loop = 0; loop < loopNum; loop++){
      /*try to make sure these files do exist */

      /*read the target file */
      if (isBunch(targetFile))
      {
          if (!readTargetBunchFile())return;
      }
      else
      {
        if (!readTargetRSFFile()) return;
      };
      /*read the source file */
      if (isBunch(sourceFile))
      {
        if (!readSourceBunchFile())return;
      }
      else
      {
        if (!readSourceRSFFile()) return;
      };

      l = tableA.size(); /* number of clusters in A */
      m = tableB.size(); /* number of clusters in B */
      if ( o != tableObj.size())
      {
        JOptionPane.showMessageDialog(this,"The total number of ojbects in source file does not equal the total number of objects in target file");
        return;
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
      if (!tagAssigment(l,m,tempA,tableObj,tableB,A)){
        return;
      }


      /*optimization*/
      //optimization(A,l);

      /* draw graph and matching */
      String showStr = maxbipartiteMatching(A,l,m,action);

      /* output the total cost */
      String resultStr = "";
      if (loop == 0 ){
        totalCost = calculateCost(l,groupscount,A,grouptags,sourceFile,targetFile);
        if (hasOption(mojoMetric)){
          /* show the mojo metric distance */
          long maxDis = maxDistanceTo(numberB,o);
          resultStr += "The MoJo distance metric is ( 1 - " + totalCost +"/" + maxDis +") = "+ mojoMetricValue(maxDis,numberB,o,totalCost) + "%\n";
        }
      }
      else{
        long totalCost2 = calculateCost(l,groupscount,A,grouptags,sourceFile,targetFile);
        if (hasOption(mojoMetric)){
          /* show the mojo metric distance */
          long maxDis = maxDistanceTo(numberB,o);
          resultStr += "The MoJo distance metric is ( 1 - " + totalCost2 +"/" + maxDis +") = "+ mojoMetricValue(maxDis,numberB,o,totalCost2) + "%\n";
        }
        resultStr += "The Mojo value is "+Math.min(totalCost,totalCost2)+"\n";
      };


      /* perform edge mojo and show status process, both needs to perform join */
      /* in practical use, only the above part is useful,
      because we only want to know the minmium number of join + move,
      we dont need to know the exact process, anyway, the following parts
      show how it works */
      if(hasOption(showStatus) || action == edgeMoJo) {
        /* output the merge opt */
        if(hasOption(showStatus)) showStr += "The process of join operations \n";
        for (int j = 0; j<m; j++){
          if ( groupscount[j] > 1 ){
            for (int i = 0; i<l; i++){
              if (A[i].getGroup() == j){
                if ( grouptags[j].getNo() != i ) {
                  grouptags[j].merge(A[i]);
                  if(hasOption(showStatus)) showStr += "join clusters "+ (String)nameA.elementAt(grouptags[j].getNo()) +" and "+(String)nameA.elementAt(i)+"\n";
                };
              }
            }
          }
        }

        if(hasOption(showStatus)){
          /* output the move opt */
          showStr += "The process of move operations \n";
          int newClusterIndex = l; /* index for newly created cluster */
          for (int j = 0; j < m; j++){
            if ( grouptags[j] != null ){
              for ( int i = 0; i <m; i ++ ){
                if ( i != j && grouptags[j].objectList[i].size() > 0){
                  showStr += "Move "+grouptags[j].objectList[i] + " from cluster "+(String)nameA.elementAt(grouptags[j].getNo());

                  if ( grouptags[i] != null ) /* the group is not empty */
                  {
                    if (grouptags[i].getNo() < nameA.size())
                      showStr += " to cluster "+(String)nameA.elementAt(grouptags[i].getNo())+"\n";
                    else{
                      if (loop == 0)
                        showStr += " to created cluster A"+(grouptags[i].getNo()+1)+"(G"+(i+1)+")\n";
                      else
                        showStr += " to created cluster B"+(grouptags[i].getNo()+1)+"(G"+(i+1)+")\n";
                    }
                  }
                  else
                  {
                    grouptags[i] = new cluster(newClusterIndex++,l,m); /* create a new Group */
                    if (loop == 0)
                      showStr += " to newly created cluster A"+(grouptags[i].getNo()+1)+"(G"+(i+1)+")\n";
                    else
                      showStr += " to newly created cluster B"+(grouptags[i].getNo()+1)+"(G"+(i+1)+")\n";
                  }
                  grouptags[j].move(i,grouptags[i]);
                }
              }
            }
          }
          /* output the processing of MoJo */
          outTextArea.append(showStr);
        };

        if (action == edgeMoJo ){
          /* calculate the additional edge cost */
          double edgeCost = 0;
          int misplacedOmniObj = 0;
          for (int j = 0; j<m; j++){
            if ( grouptags[j] != null)
            {
              if ( hasOption(omniPresent)){
                edgeCost += grouptags[j].edgeCost(tableR,grouptags,omniVector);
                misplacedOmniObj += grouptags[j].getMisplacedOmnipresentObjects();
              }
              else
                edgeCost += grouptags[j].edgeCost(tableR,grouptags,null);
            }
          }
          resultStr = "The additional cost of edge is "+edgeCost+"\n"+ resultStr;
          if ( hasOption(omniPresent))
            resultStr = "The number of misplaced omnipresent objects is "+misplacedOmniObj+"\n"+ resultStr;
        }
      }
      /*output the result of MoJo */
      outTextArea.append(resultStr);

      /* re init the variable */
      if ( loopNum == doubleDirection && loop == 0){
        /* this hashtable is used to store the relation between object and cluster in B */
        tableObj = new Hashtable();
        /* countA & countB is used to calculate the different number of clusters in A & B */
        tableA = new Hashtable();
        tableB = new Hashtable();

        /* use for store the name of each items */
        nameA = new Vector();
        nameB = new Vector();

        /* This vector is used to store all the objects and cluster in A */
        tempA  = new Vector();
        tempB  = new Vector();
        /* use for store the number of ojbects in each cluster in partition B
        this is use in calculate the maxdistance from partition B */
        numberB = new Vector();

        l = 0;    /* number of clusters in A */
        m = 0;    /* number of clusters in B */
        o = 0;   /* number of total objects */
        A = null;    /* A */

        /* record the capicity of each group, if the group is empty
        ,the count is zero, otherwise >= 1 */
        groupscount = null;

        grouptags = null; /* every none empty group have a tag point to a cluser in A */

        /*change file name */
        /* the format is MoJo [argument] A B
        in the first loop A = sourceFile B = targetFile
        in the second loop change A = targetFile B = sourceFile */
        sourceFile = targetjTextField.getText();
        targetFile = sourcejTextField.getText();


      };

    };

  }
    /* read the source file, put all the objects into vector tempA,tempA.elementOf(i)
    will be the cluster A[i].When we want to determine a objects tag, just use tableObj.get
    to determine which cluster in B the current objects belongs in */
  boolean readSourceBunchFile(){
    try
    {
      br_s = new BufferedReader ( new FileReader(sourceFile) );
    }
    catch ( FileNotFoundException e)
    {
      JOptionPane.showMessageDialog(this,"source file "+sourceFile+" not found");
      return false;
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
      JOptionPane.showMessageDialog(this,"Error reading data file");
      return false;
    };
    try {
      br_s.close();
    }
    catch ( IOException ef)
    {
      JOptionPane.showMessageDialog(this,ef.getMessage());
      return false;
    };
    return true;
  }

  boolean readSourceRSFFile(){
    try
    {
      br_s = new BufferedReader ( new FileReader(sourceFile) );
    }
    catch ( FileNotFoundException ef)
    {
      JOptionPane.showMessageDialog(this,"source file "+sourceFile+" not found");
      return false;
    };

    try
    {
      for (String str_s = br_s.readLine(); str_s != null; str_s = br_s.readLine())
      {

        StringTokenizer st = new StringTokenizer(str_s);
        if ( st.countTokens() != 3) {
          JOptionPane.showMessageDialog(this,"wrong format in one line: doesn't contain 3 field");
          return false;
        };

        if ( !st.nextToken().toLowerCase().equals("contain")) {
          JOptionPane.showMessageDialog(this,"wrong format in one line: the first field must be contain");
          return false;
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
    catch (IOException ef)
    {
      JOptionPane.showMessageDialog(this,"Error reading data file");
      return false;
    };
    try {
      br_s.close();
    }
    catch ( IOException ef)
    {
      JOptionPane.showMessageDialog(this,ef.getMessage());
      return false;
    };
    return true;

  }
  /* this is similiar to readTargetBunchFile, but it is used for omnipresent node detection */
  boolean readAuthorativePatitionBunchFile(String fileName){
      /* read the target file, put all the objects into hash table tableAuthorativePatition_Obj,
      then we can get a object's tag easily when we read the source file, we also
      count the size of cluster B using hashtalbe tableB */
      BufferedReader br_a;
      try
      {
        br_a = new BufferedReader ( new FileReader(fileName) );
      }
      catch ( FileNotFoundException e)
      {
        JOptionPane.showMessageDialog(this,"target partition file "+fileName+" not found");
        return false;
      };
      try{
        tableAuthorativePatition_Obj = new Hashtable();
        for (String str_t = br_a.readLine(); str_t != null; str_t = br_a.readLine())
        {
          int equalMark = str_t.indexOf("=");
          String strClusterB = str_t.substring(0,equalMark).trim();
          String objList = str_t.substring(equalMark+1,str_t.length()).trim();
          StringTokenizer st = new StringTokenizer(objList,",");
          int objNumber = st.countTokens();

          for (int i = 0; i < objNumber; i++)
          {
            String obj = st.nextToken().trim();
            /* just let we know which object is in wich cluster */
            tableAuthorativePatition_Obj.put(obj, strClusterB);
          }
        }
      }
      catch (IOException e)
      {
        JOptionPane.showMessageDialog(this,"Error reading data file");
        return false;
      };
      try {
        br_a.close();
      }
      catch ( IOException e)
      {
        JOptionPane.showMessageDialog(this,e.getMessage());
        return false;
      };
      return true;


  }
  /* read the target file, put all the objects into hash table tableObj,
  then we can get a object's tag easily when we read the source file, we also
  count the size of cluster B using hashtalbe tableB */
   /*read target file */
   boolean readTargetBunchFile(){
      /* read the target file, put all the objects into hash table tableObj,
      then we can get a object's tag easily when we read the source file, we also
      count the size of cluster B using hashtalbe tableB */
      try
      {
        br_t = new BufferedReader ( new FileReader(targetFile) );
      }
      catch ( FileNotFoundException e)
      {
        JOptionPane.showMessageDialog(this,"target file "+targetFile+" not found");
        return false;
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
          tempB.addElement(new Vector());

          /* the first time, it contain only one object, i.e., the current object */
          numberB.addElement(new Integer(objNumber));
          tableB.put(strClusterB,new Integer(index));

          for (int i = 0; i < objNumber; i++)
          {
            String obj = st.nextToken().trim();
            ((Vector)tempB.elementAt(index)).addElement(obj);
            tableObj.put(obj, strClusterB);
          }
        }
      }
      catch (IOException e)
      {
        JOptionPane.showMessageDialog(this,"Error reading data file");
        return false;
      };
      try {
        br_t.close();
      }
      catch ( IOException e)
      {
        JOptionPane.showMessageDialog(this,e.getMessage());
        return false;
      };
      return true;

   }
  /* this is similiar to readTargetBunchFile, but it is used for omnipresent node detection */
  boolean readAuthorativePatitionRSFFile(String fileName){
      BufferedReader br_a;
      try
      {
        br_a = new BufferedReader ( new FileReader(fileName) );
      }
      catch ( FileNotFoundException e)
      {
        JOptionPane.showMessageDialog(this,"target partition file "+fileName+" not found");
        return false;
      };

      try{
      	tableAuthorativePatition_Obj = new Hashtable();
        for (String str_t = br_a.readLine(); str_t != null; str_t = br_a.readLine())
        {
	        StringTokenizer st = new StringTokenizer(str_t);
	        if ( st.countTokens() != 3) {
	          JOptionPane.showMessageDialog(this,"wrong format in one line: doesn't contain 3 field");
	          return false;
	        };

	        if ( !st.nextToken().toLowerCase().equals("contain")) {
	          JOptionPane.showMessageDialog(this,"wrong format in one line: the first field must be contain");
	          return false;
	        };
	        String strClusterB = st.nextToken();
        	/* transfer "A_4" to A_4 */

           int first_quote_index = strClusterB.indexOf("\"");
	        if ( first_quote_index == 0 && strClusterB.indexOf("\"",first_quote_index+1) == strClusterB.length()-1)
        	  strClusterB = strClusterB.substring(first_quote_index+1,strClusterB.length()-1);

	        String obj = st.nextToken();
           tableAuthorativePatition_Obj.put(obj, strClusterB);

        }
      }
      catch (IOException e)
      {
        JOptionPane.showMessageDialog(this,"Error reading data file");
        return false;
      };
      try {
        br_a.close();
      }
      catch ( IOException e)
      {
        JOptionPane.showMessageDialog(this,e.getMessage());
        return false;
      };
      return true;


  }

  boolean readTargetRSFFile(){
    try
    {
      br_t = new BufferedReader ( new FileReader(targetFile) );
    }
    catch ( FileNotFoundException ef)
    {
      JOptionPane.showMessageDialog(this,"target file "+targetFile+" not found");
      return false;
    };

    try{
      for (String str_t = br_t.readLine(); str_t != null; str_t = br_t.readLine())
      {

        StringTokenizer st = new StringTokenizer(str_t);
        if ( st.countTokens() != 3) {
          JOptionPane.showMessageDialog(this,"wrong format in one line: doesn't contain 3 field");
          return false;
        };

        if ( !st.nextToken().toLowerCase().equals("contain")) {
          JOptionPane.showMessageDialog(this,"wrong format in one line: the first field must be contain");
          return false;
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
          tempB.add(new Vector());
          ((Vector)tempB.elementAt(index)).addElement(obj);
          tableB.put(strClusterB,new Integer(index));

        }
        else
        {
          index = ((Integer)objectIndex).intValue();
          /* how many objects there are in the current cluster B_index, and we add one then store back*/
          int cur_number = 1 + ((Integer)numberB.elementAt(index)).intValue();
          /* set it back */
          numberB.setElementAt(new Integer(cur_number),index);
          ((Vector)tempB.elementAt(index)).addElement(obj);
        }

        tableObj.put(obj, strClusterB);
      };
    }
    catch (IOException ef)
    {
      JOptionPane.showMessageDialog(this,"Error reading data file");
      return false;
    };

    try {
      br_t.close();
    }
    catch ( IOException ef)
    {
      JOptionPane.showMessageDialog(this,ef.getMessage());
      return false;
    };
    return true;

  }
  /* init the variable */
  void init_var(){

    outTextArea.setText("");
    /* this hashtable is used to store the relation between object and cluster in B */
    /* set the file name */
    sourceFile = sourcejTextField.getText();
    targetFile = targetjTextField.getText();
    relationFile = relationjTextField.getText();

    /*table of object*/
    tableObj = new Hashtable();
    /* countA & countB is used to calculate the different number of clusters in A & B */
    tableA = new Hashtable();
    tableB = new Hashtable();
    /* use for store the name of each items */
    nameA = new Vector();
    nameB = new Vector();
    /* use for store the number of ojbects in each cluster in partition B
     this is use in calculate the maxdistance from partition B */
    numberB = new Vector();
    /* This vector is used to store all the objects and cluster in A */
    tempA  = new Vector();
    tempB  = new Vector();


    l = 0;    /* number of clusters in A */
    m = 0;    /* number of clusters in B */
    o = 0;   /* number of total objects */
    totalCost = 0; /*total cost of Mojo */
    A = null;    /* A */

    /* record the capicity of each group, if the group is empty
    ,the count is zero, otherwise >= 1 */
    groupscount = null;

    /* after join operations, each group will have only one cluster left,
    we use grouptags[i] to indicate the remain cluster in group i*/
    grouptags = null; /* every none empty group have a tag point to a cluser in A */
  }
  boolean isBunch(String file){
    int dot = file.lastIndexOf(".");
    if (dot < 0 ) return false;
    String extend = file.substring(dot+1).trim();
    if (extend.equalsIgnoreCase("bunch")) return true;
    return false;
  }



}
