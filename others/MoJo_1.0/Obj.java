import java.awt.*;

/**
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2002</p>
 * <p>Company: </p>
 * @author Zhihua Wen
 * @version 1.0
 */

public class Obj {
  /*whether the current object is been moved or not */
  boolean isMoving = false;
  private final int width = 24;
  private final int height = 24;
  private String name;
  private int x,y;
  private int oldx,oldy;

  public void setX(int x){
    oldx = x;
    this.x = x;
  }
  public int getX(){
    return x;
  }
  public void setY(int y){
    oldy = y;
    this.y = y;
  }
  public int getY(){
    return y;
  }

  public Obj(String name) {
    this.name = name;
  }
  public String getName(){
    return name;
  }
  public void setPos(int x, int y){
    oldx = x;
    oldy = y;
    this.x = x;
    this.y = y;
  }
  public void paint(ClusterAni c, Graphics g,int x,int y){
    setPos(x,y);
    g.clearRect(oldx,oldy,width,height);
    paint(c,g);
  }
  public void paint(ClusterAni c,Graphics g){

    g.clearRect(x,y,width,height);

    if ( isMoving && c.isleft )
      g.setColor(Color.red);
    else
      g.setColor(Color.black);
    g.drawOval(x,y,width,height);

    g.setColor(Color.black);
    int strlen = (g.getFontMetrics()).stringWidth(name);
    int strHeight = (g.getFontMetrics()).getHeight();
    g.drawString(name,x+(width-strlen)/2,y+(height-strHeight)/2+3* strHeight/4);
    g.setColor(Color.black);
  }
}