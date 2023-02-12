import java.awt.*;
import java.awt.event.*;
import java.applet.*;
import javax.swing.*;
import javax.swing.border.*;
import java.util.*;

// for dealing with the start stop button
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

// for dealing with the delay slider
import java.awt.event.AdjustmentListener;
import java.awt.event.AdjustmentEvent;

/**
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2002</p>
 * <p>Company: </p>
 * @author unascribed
 * @version 1.0
 */

public class mojoAnimation extends JFrame
                          implements Runnable,ActionListener,AdjustmentListener {

  Thread runner = null;	// The thread that is displaying the text
  boolean threadSuspended;	// True when thread suspended (via mouse click)
  boolean isRandom = false; // is this animation based on random clusters or not

  private Button startButton = new Button("assign Tags");
  private Button resetButton = new Button("        Reset         ");
  private Label label1 = new Label("Clusters in A");
  private Label label2 = new Label("Clusters in B");
  private Label label3 = new Label("Total objects");

  JTextField fieldA = new JTextField("3",3);
  JTextField fieldB = new JTextField("3",3);
  JTextField fieldO = new JTextField("15",3);

  private Label label4 = new Label("DELAY: 1000 milliseconds");
  JScrollBar DelaySlider = new JScrollBar(Scrollbar.HORIZONTAL,100, 10, 0, 500);
  public Label stateLabel = new Label("");
  JPanel inputPanel = new JPanel(); // Panel
  private drawArea drawArea;
  JFrame parent;

  //Initialize the applet from the file
  public mojoAnimation(JFrame frame) {
    isRandom = true;
    parent = frame;
    JPanel contentPane = (JPanel) this.getContentPane();
    threadSuspended = false;
    setFont(new Font("Dialog",java.awt.Font.PLAIN,12));
    BorderLayout border = new BorderLayout();
    GridBagLayout gridBag = new GridBagLayout();
    GridBagConstraints c = new GridBagConstraints();
    inputPanel.setLayout(gridBag);
    this.getContentPane().setLayout(border);  // our main layout the border layout.

    drawArea = new drawArea(this);
    contentPane.add("Center",drawArea);
    c.fill = GridBagConstraints.HORIZONTAL;
    c.weightx = 1.0;
    c.weighty = 0.0;
    startButton.setLabel("Create Clusters");
    gridBag.setConstraints(label1, c);
    inputPanel.add(label1);
    gridBag.setConstraints(fieldA, c);
    inputPanel.add(fieldA);
    gridBag.setConstraints(label2, c);
    inputPanel.add(label2);
    gridBag.setConstraints(fieldB, c);
    inputPanel.add(fieldB);
    gridBag.setConstraints(label3, c);
    inputPanel.add(label3);
    gridBag.setConstraints(fieldO, c);
    inputPanel.add(fieldO);
    c.gridwidth = GridBagConstraints.RELATIVE; //end row
    gridBag.setConstraints(startButton, c);
    inputPanel.add(startButton);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    gridBag.setConstraints(resetButton, c);
    inputPanel.add(resetButton);
    inputPanel.add(label4);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    gridBag.setConstraints(DelaySlider, c);
    inputPanel.add(DelaySlider);
    contentPane.add("North",inputPanel);
    contentPane.add("South",stateLabel);
    validate();
    startButton.addActionListener(this);
    resetButton.addActionListener(this);
    DelaySlider.addAdjustmentListener(this);
  }
  //Initialize the applet from the file
  public mojoAnimation(JFrame frame, Vector tempA, Vector tempB,long o) {
    isRandom = false;
    parent = frame;
    JPanel contentPane = (JPanel) this.getContentPane();
    threadSuspended = false;
    setFont(new Font("Dialog",java.awt.Font.PLAIN,12));
    BorderLayout border = new BorderLayout();
    GridBagLayout gridBag = new GridBagLayout();
    GridBagConstraints c = new GridBagConstraints();
    inputPanel.setLayout(gridBag);
    this.getContentPane().setLayout(border);  // our main layout the border layout.

    drawArea = new drawArea(this,tempA,tempB,o);
    contentPane.add("Center",drawArea);
    c.fill = GridBagConstraints.HORIZONTAL;
    c.weightx = 1.0;
    c.weighty = 0.0;

    gridBag.setConstraints(label1, c);
    inputPanel.add(label1);
    gridBag.setConstraints(fieldA, c);
    fieldA.setEditable(false);
    inputPanel.add(fieldA);
    gridBag.setConstraints(label2, c);
    inputPanel.add(label2);
    gridBag.setConstraints(fieldB, c);
    fieldB.setEditable(false);
    inputPanel.add(fieldB);
    gridBag.setConstraints(label3, c);
    inputPanel.add(label3);
    gridBag.setConstraints(fieldO, c);
    fieldO.setEditable(false);
    inputPanel.add(fieldO);
    c.gridwidth = GridBagConstraints.RELATIVE; //end row
    gridBag.setConstraints(startButton, c);
    inputPanel.add(startButton);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    gridBag.setConstraints(resetButton, c);
    inputPanel.add(resetButton);
    inputPanel.add(label4);
    c.gridwidth = GridBagConstraints.REMAINDER; //end row
    gridBag.setConstraints(DelaySlider, c);
    inputPanel.add(DelaySlider);
    contentPane.add("North",inputPanel);
    contentPane.add("South",stateLabel);
    validate();
    startButton.addActionListener(this);
    resetButton.addActionListener(this);
    DelaySlider.addAdjustmentListener(this);
  }
  //Component initialization

  //Start the applet
  public void start() {
      runner = new Thread(this);
      runner.start();
  }
  //Stop the applet
  public synchronized void stop() {
      runner = null;
      if (threadSuspended) {
          threadSuspended = false;
          notify();
      }
  }
  public void run() {
      Thread me = Thread.currentThread();
      while (runner == me) {
          try {
              Thread.sleep(100);
              synchronized(this) {
                  while (threadSuspended) {
                      wait();
                  }
              }
          } catch (InterruptedException e){
          }
          repaint();
      }
  }

  protected void processWindowEvent(WindowEvent e) {
    super.processWindowEvent(e);
    if (e.getID() == WindowEvent.WINDOW_CLOSING) {
      parent.setEnabled(true);
      parent.setVisible(true);

    }
  }

  public void actionPerformed(ActionEvent e) {

    if (e.getSource() == resetButton){
      drawArea.reset();
      if (isRandom)
        startButton.setLabel("Create Clusters");
      else
        startButton.setLabel("assign Tags");
      return;

    }
    if (e.getSource() == startButton ){
      if ( startButton.getLabel().equals("Create Clusters")){
        drawArea.randomCreate();
        startButton.setLabel("assign Tags");
        return;
      }
      if ( startButton.getLabel().equals("assign Tags")){
        if (drawArea.calculateTag()){
          startButton.setLabel("generate Graph");
          return;
        }
        else{
          JOptionPane.showMessageDialog(this,"pleae close the window");
          dispose();
          parent.setEnabled(true);
          parent.setVisible(true);
        }
      }
      if ( startButton.getLabel().equals("generate Graph")){
        drawArea.generateGraph();
        startButton.setLabel("Matching");
        return;
      }
      if ( startButton.getLabel().equals("Matching")){
        startButton.setEnabled(false);
        drawArea.Matching();
        startButton.setLabel("Assign Groups");
        startButton.setEnabled(true);
        return;
        //}
      }

      if ( startButton.getLabel().equals("Assign Groups") ){
        drawArea.assignGroup();
        startButton.setLabel("Join operations");
        return;
      }

      if ( startButton.getLabel().equals("Join operations")){
        startButton.setEnabled(false);
        drawArea.join();
        startButton.setLabel("Move operations");
        startButton.setEnabled(true);
        return;
      }

      if ( startButton.getLabel().equals("Move operations")){
        startButton.setEnabled(false);
        drawArea.move();
        startButton.setLabel("END/Reset");
        startButton.setEnabled(true);
        return;
      }

      if ( startButton.getLabel().equals("END/Reset")){
        drawArea.reset();
        if (isRandom)
          startButton.setLabel("Create Clusters");
        else
          startButton.setLabel("assign Tags");
        return;
      }


    }

  }

  public void adjustmentValueChanged(AdjustmentEvent e) {
    if ( e.getSource() == DelaySlider){
      drawArea.delay = DelaySlider.getValue();
      label4.setText("DELAY: "+String.valueOf(10*DelaySlider.getValue()) +
                  " milliseconds");
    }

  }

  public mojoAnimation() {
    try {
      jbInit();
    }
    catch(Exception e) {
      e.printStackTrace();
    }
  }
  private void jbInit() throws Exception {
    this.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
  }

}
