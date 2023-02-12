import java.awt.*;
import java.awt.event.*;
import java.net.*;
import java.io.*;
import javax.swing.*;
import javax.swing.border.*;

/**
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2002</p>
 * <p>Company: </p>
 * @author Zhihua Wen
 * @version 1.0
 */

public class MoJoFrame_AboutBox extends JDialog implements ActionListener {

  private JPanel panel1 = new JPanel();
  private JPanel panel2 = new JPanel();
  private JPanel insetsPanel1 = new JPanel();
  private JPanel insetsPanel2 = new JPanel();
  private JPanel insetsPanel3 = new JPanel();
  private JButton button1 = new JButton();
  private JLabel imageLabel = new JLabel();
  private JLabel label1 = new JLabel();
  private JLabel label2 = new JLabel();
  private JLabel label3 = new JLabel();
  private JLabel label4 = new JLabel();
  private BorderLayout borderLayout1 = new BorderLayout();
  private BorderLayout borderLayout2 = new BorderLayout();
  private FlowLayout flowLayout1 = new FlowLayout();
  private GridLayout gridLayout1 = new GridLayout();
  private String product = "MoJo Application";
  private String version = "1.2";
  private String copyright = "Copyright (c) 2003";
  private String comments = "";
  private JPanel jPanel1 = new JPanel();
  private Border border1;
  private JLabel jLabel1 = new JLabel();
  private JLabel jLabel2 = new JLabel();
  private JLabel jLabel3 = new JLabel();
  private JLabel jLabel4 = new JLabel();
  private JLabel jLabel5 = new JLabel();
  private JLabel jLabel6 = new JLabel();
  private JLabel jLabel7 = new JLabel();
  private JLabel jLabel8 = new JLabel();
  private GridBagLayout gridBagLayout1 = new GridBagLayout();
  public MoJoFrame_AboutBox(Frame parent) {
    super(parent);
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
    //imageLabel.setIcon(new ImageIcon(MoJoFrame_AboutBox.class.getResource("[Your Image]")));
    border1 = BorderFactory.createCompoundBorder(new EtchedBorder(EtchedBorder.RAISED,Color.white,new Color(148, 145, 140)),BorderFactory.createEmptyBorder(5,5,5,5));
    this.setTitle("About");
    panel1.setLayout(borderLayout1);
    panel2.setLayout(borderLayout2);
    insetsPanel1.setLayout(flowLayout1);
    insetsPanel2.setLayout(flowLayout1);
    insetsPanel2.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
    gridLayout1.setRows(4);
    gridLayout1.setColumns(1);
    label1.setText(product);
    label2.setText(version);
    label3.setText(copyright);
    label4.setText(comments);
    insetsPanel3.setLayout(gridLayout1);
    insetsPanel3.setBorder(BorderFactory.createEmptyBorder(10, 60, 10, 10));
    button1.setText("Ok");
    button1.addActionListener(this);
    jPanel1.setBorder(border1);
    jPanel1.setMinimumSize(new Dimension(100, 100));
    jPanel1.setLayout(gridBagLayout1);
    jLabel1.setToolTipText("");
    jLabel1.setText("                      Running method:                       ");
    jLabel2.setText("                     java -jar mojo.jar                     ");
    jLabel3.setText("java -jar mojo.jar [-a -s -as -v - m -e] a.rsf b.rsf [r.rsf]");
    jLabel4.setText("       Or extract the package, running the class file       ");
    jLabel5.setText("                        java MoJo                           ");
    jLabel6.setText("     java MoJo [-a -s -as -v - m -e] a.rsf b.rsf [r.rsf]    ");
    jLabel7.setText("           For help or documentation, please visit          ");
    jLabel8.setText("                http://www.cs.yorku.ca/~bil                 ");
    jLabel8.addMouseMotionListener(new MoJoFrame_AboutBox_jLabel8_mouseMotionAdapter(this));
    jLabel8.addMouseListener(new MoJoFrame_AboutBox_jLabel8_mouseAdapter(this));
    jLabel8.setForeground(Color.blue);
    insetsPanel2.add(imageLabel, null);
    panel2.add(insetsPanel2, BorderLayout.WEST);
    this.getContentPane().add(panel1, null);
    insetsPanel3.add(label1, null);
    insetsPanel3.add(label2, null);
    insetsPanel3.add(label3, null);
    insetsPanel3.add(label4, null);
    panel1.add(jPanel1,  BorderLayout.CENTER);
    GridBagConstraints c = new GridBagConstraints();
    c.gridwidth = GridBagConstraints.REMAINDER;
    c.fill = GridBagConstraints.HORIZONTAL;
    c.weightx = 1.0;
    c.weighty = 0.0;

    jPanel1.add(jLabel1,c);
    jPanel1.add(jLabel2,c);
    jPanel1.add(jLabel3,c);
    jPanel1.add(jLabel4,c);
    jPanel1.add(jLabel5,c);
    jPanel1.add(jLabel6,c);
    jPanel1.add(jLabel7,c);
    jPanel1.add(jLabel8,c);
    panel2.add(insetsPanel3, BorderLayout.CENTER);
    insetsPanel1.add(button1, null);
    panel1.add(insetsPanel1, BorderLayout.SOUTH);
    panel1.add(panel2, BorderLayout.NORTH);
    setResizable(false);
  }
  //Overridden so we can exit when window is closed
  protected void processWindowEvent(WindowEvent e) {
    if (e.getID() == WindowEvent.WINDOW_CLOSING) {
      cancel();
    }
    super.processWindowEvent(e);
  }
  //Close the dialog
  void cancel() {
    dispose();
  }
  //Close the dialog on a button event
  public void actionPerformed(ActionEvent e) {
    if (e.getSource() == button1) {
      cancel();
    }
  }

  void jLabel8_mouseClicked(MouseEvent e) {

    try{
      URL url = new URL(jLabel8.getText());
      try{

        String link = "http://www.cs.yorku.ca/~bil";
        String osName = System.getProperty("os.name").toLowerCase();

        if (osName.indexOf("windows") > - 1){
          if (osName.indexOf("nt") > -1 || osName.indexOf("2000") > -1 || osName.indexOf("xp")>-1)
            Runtime.getRuntime().exec("cmd /c start "+link);
          else
            Runtime.getRuntime().exec("command /c start "+link);
        }
        else
          Runtime.getRuntime().exec("netscape "+link);

      }
      catch (IOException ie){};
    }
    catch (MalformedURLException me){};
  }

  void jLabel8_mouseMoved(MouseEvent e) {

  }
}

class MoJoFrame_AboutBox_jLabel8_mouseAdapter extends java.awt.event.MouseAdapter {
  MoJoFrame_AboutBox adaptee;

  MoJoFrame_AboutBox_jLabel8_mouseAdapter(MoJoFrame_AboutBox adaptee) {
    this.adaptee = adaptee;
  }
  public void mouseClicked(MouseEvent e) {
    adaptee.jLabel8_mouseClicked(e);
  }
}

class MoJoFrame_AboutBox_jLabel8_mouseMotionAdapter extends java.awt.event.MouseMotionAdapter {
  MoJoFrame_AboutBox adaptee;

  MoJoFrame_AboutBox_jLabel8_mouseMotionAdapter(MoJoFrame_AboutBox adaptee) {
    this.adaptee = adaptee;
  }
  public void mouseMoved(MouseEvent e) {
    adaptee.jLabel8_mouseMoved(e);
  }
}
