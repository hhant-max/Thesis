import javax.swing.UIManager;
import java.awt.*;

/**
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2002</p>
 * <p>Company: </p>
 * @author unascribed
 * @version 1.0
 */

public class MoJo {

  //Construct the application
  public MoJo() {
  }
  //Main method
  public static void main(String[] args) {
    if ( args.length == 0 ){
      MoJoFrame frame = new MoJoFrame();
      //Validate frames that have preset sizes
      //Pack frames that have useful preferred size info, e.g. from their layout
      //  frame.pack();
        frame.validate();
      //Center the window
      Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
      Dimension frameSize = frame.getSize();
      if (frameSize.height > screenSize.height) {
        frameSize.height = screenSize.height;
      }
      if (frameSize.width > screenSize.width) {
        frameSize.width = screenSize.width;
      }
      frame.setLocation((screenSize.width - frameSize.width) / 2, (screenSize.height - frameSize.height) / 2);
      frame.setSize(frameSize);
      frame.setVisible(true);

      try {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
      }
      catch(Exception e) {
        e.printStackTrace();
      }
      //new MoJo();
    }
    else{
      new MoJoCommand().main(args);
    }
  }
}
