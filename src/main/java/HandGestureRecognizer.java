import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.InputEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;


public class HandGestureRecognizer extends JPanel {

    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private JFrame frame = new JFrame("Hand gestures recognition");
    private JLabel lab = new JLabel();
    private static String WAITING_FOR_ACTION_STRING = "Waiting for action...";

    private static Point last = new Point();
    private static boolean close = false;
    private static boolean act = false;
    private static long current = 0;
    private static long prev = 0;
    private static boolean start = false;

    /**
     * Create the panel.
     */
    public HandGestureRecognizer() {

    }

    public void setframe(final VideoCapture webcam) {
        frame.setSize(1024, 768);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        frame.getContentPane().add(lab);
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.out.println("Closed");
                close = true;
                webcam.release();
                e.getWindow().dispose();
            }
        });
    }

    public void frametolabel(Mat matframe) {
        MatOfByte cc = new MatOfByte();
        Highgui.imencode(".JPG", matframe, cc);
        byte[] bytes = cc.toArray();
        InputStream ss = new ByteArrayInputStream(bytes);
        try {
            BufferedImage image = ImageIO.read(ss);
            lab.setIcon(new ImageIcon(image));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double calculateAngle(Point P1, Point P2, Point P3) {
        double angle2 = 0;
        Point v1 = new Point();
        Point v2 = new Point();
        v1.x = P3.x - P1.x;
        v1.y = P3.y - P1.y;
        v2.x = P3.x - P2.x;
        v2.y = P3.y - P2.y;
        double dotproduct = (v1.x * v2.x) + (v1.y * v2.y);
        double length1 = Math.sqrt((v1.x * v1.x) + (v1.y * v1.y));
        double length2 = Math.sqrt((v2.x * v2.x) + (v2.y * v2.y));
        double angle = Math.acos(dotproduct / (length1 * length2));
        angle2 = angle * 180 / Math.PI;

        return angle2;
    }

    public Mat filterColorHsv(int h, int s, int v, int h1, int s1, int v1, Mat immagine) {
        Mat mod = new Mat();
        if (immagine != null) {
            Core.inRange(immagine, new Scalar(h, s, v), new Scalar(h1, s1, v1), mod);
        } else {
            System.out.println("Error with image!");
        }
        return mod;
    }

    public Mat detectSkin(Mat orig) {
        Mat mask = new Mat();
        Mat result = new Mat();
        Core.inRange(orig, new Scalar(0, 0, 0), new Scalar(30, 30, 30), result);
        Imgproc.cvtColor(orig, mask, Imgproc.COLOR_BGR2HSV);
        for (int i = 0; i < mask.size().height; i++) {
            for (int j = 0; j < mask.size().width; j++) {
                if (mask.get(i, j)[0] < 19 || mask.get(i, j)[0] > 150
                        && mask.get(i, j)[1] > 25 && mask.get(i, j)[1] < 220) {

                    result.put(i, j, 255, 255, 255);

                } else {
                    result.put(i, j, 0, 0, 0);
                }
            }

        }


        return result;

    }

    public Mat morphFilter(int kd, int ke, Mat img) {
        Mat mod = new Mat();
        Imgproc.erode(img, mod, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(ke, ke)));
        //Imgproc.erode(mod, mod, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(ke,ke)));
        Imgproc.dilate(mod, mod, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kd, kd)));
        return mod;

    }

    public List<MatOfPoint> searchArea(Mat original, Mat img, boolean draw, boolean flag, int pixelFilter) {
        List<MatOfPoint> contours = new LinkedList<MatOfPoint>();
        List<MatOfPoint> contoursbig = new LinkedList<MatOfPoint>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE, new Point(0, 0));

        for (int i = 0; i < contours.size(); i++) {
            if (contours.get(i).size().height > pixelFilter) {
                contoursbig.add(contours.get(i));
                if (draw && !flag)
                    Imgproc.drawContours(original, contours, i, new Scalar(0, 255, 0), 2, 8, hierarchy, 0, new Point());
            }

            if (flag && !draw)
                Imgproc.drawContours(original, contours, i, new Scalar(0, 255, 255), 2, 8, hierarchy, 0, new Point());

        }
        return contoursbig;
    }

    public List<Point> countourList(Mat img, int pixelFilter) {
        List<MatOfPoint> contours = new LinkedList<MatOfPoint>();
        List<MatOfPoint> contoursbig = new LinkedList<MatOfPoint>();
        List<Point> points = new LinkedList<Point>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE, new Point(0, 0));

        for (int i = 0; i < contours.size(); i++) {
            if (contours.get(i).size().height > pixelFilter) {
                contoursbig.add(contours.get(i));
            }

        }
        if (contoursbig.size() > 0) {

            points = contoursbig.get(0).toList();

        }
        return points;
    }

    public List<Point> defect(Mat img, List<MatOfPoint> contours, boolean draw, int depthThreshold) {
        List<Point> defects = new LinkedList<Point>();

        for (int i = 0; i < contours.size(); i++) {
            MatOfInt hull_ = new MatOfInt();
            MatOfInt4 convexityDefects = new MatOfInt4();

            Imgproc.convexHull(contours.get(i), hull_);

            if (hull_.size().height >= 4) {


                Imgproc.convexityDefects(contours.get(i), hull_, convexityDefects);

                List<Point> pts = new ArrayList<Point>();
                MatOfPoint2f pr = new MatOfPoint2f();
                Converters.Mat_to_vector_Point(contours.get(i), pts);

                pr.create((int) (pts.size()), 1, CvType.CV_32S);
                pr.fromList(pts);
                if (pr.height() > 10) {
                    RotatedRect r = Imgproc.minAreaRect(pr);
                    Point[] rect = new Point[4];
                    r.points(rect);

                    Core.line(img, rect[0], rect[1], new Scalar(0, 100, 0), 2);
                    Core.line(img, rect[0], rect[3], new Scalar(0, 100, 0), 2);
                    Core.line(img, rect[1], rect[2], new Scalar(0, 100, 0), 2);
                    Core.line(img, rect[2], rect[3], new Scalar(0, 100, 0), 2);
                    Core.rectangle(img, r.boundingRect().tl(), r.boundingRect().br(), new Scalar(50, 50, 50));
                }

                int[] buff = new int[4];
                int[] zx = new int[1];
                int[] zxx = new int[1];
                for (int i1 = 0; i1 < hull_.size().height; i1++) {
                    if (i1 < hull_.size().height - 1) {
                        hull_.get(i1, 0, zx);
                        hull_.get(i1 + 1, 0, zxx);
                    } else {
                        hull_.get(i1, 0, zx);
                        hull_.get(0, 0, zxx);
                    }
                    if (draw)
                        Core.line(img, pts.get(zx[0]), pts.get(zxx[0]), new Scalar(140, 140, 140), 2);
                }


                for (int i1 = 0; i1 < convexityDefects.size().height; i1++) {
                    convexityDefects.get(i1, 0, buff);
                    if (buff[3] / 256 > depthThreshold) {
                        if (pts.get(buff[2]).x > 0 && pts.get(buff[2]).x < 1024 && pts.get(buff[2]).y > 0 && pts.get(buff[2]).y < 768) {
                            defects.add(pts.get(buff[2]));
                            Core.circle(img, pts.get(buff[2]), 6, new Scalar(0, 255, 0));
                            if (draw)
                                Core.circle(img, pts.get(buff[2]), 6, new Scalar(0, 255, 0));

                        }
                    }
                }
                if (defects.size() < 3) {
                    int dim = pts.size();
                    Core.circle(img, pts.get(0), 3, new Scalar(0, 255, 0), 2);
                    Core.circle(img, pts.get(0 + dim / 4), 3, new Scalar(0, 255, 0), 2);
                    defects.add(pts.get(0));
                    defects.add(pts.get(0 + dim / 4));


                }
            }
        }
        return defects;
    }

    public Point centerPalm(Mat img, List<Point> defect) {
        MatOfPoint2f pr = new MatOfPoint2f();
        Point center = new Point();
        float[] radius = new float[1];
        pr.create((int) (defect.size()), 1, CvType.CV_32S);
        pr.fromList(defect);

        if (pr.size().height > 0) {
            start = true;
            Imgproc.minEnclosingCircle(pr, center, radius);

            //Core.circle(img, center,(int) radius[0], new Scalar(255,0,0));
            //  Core.circle(img, center, 3, new Scalar(0,0,255),4);
        } else {
            start = false;
        }
        return center;

    }

    public List<Point> process(Mat img, List<Point> points, Point center) {
        List<Point> pointList = new LinkedList<Point>();
        List<Point> processedPoints = new LinkedList<Point>();
        int interval = 55;
        for (int j = 0; j < points.size(); j++) {
            Point prev = new Point();
            Point vertex = new Point();
            Point next = new Point();
            vertex = points.get(j);
            if (j - interval > 0) {

                prev = points.get(j - interval);
            } else {
                int a = interval - j;
                prev = points.get(points.size() - a - 1);
            }
            if (j + interval < points.size()) {
                next = points.get(j + interval);
            } else {
                int a = j + interval - points.size();
                next = points.get(a);
            }

            Point v1 = new Point();
            Point v2 = new Point();
            v1.x = vertex.x - next.x;
            v1.y = vertex.y - next.y;
            v2.x = vertex.x - prev.x;
            v2.y = vertex.y - prev.y;
            double dotproduct = (v1.x * v2.x) + (v1.y * v2.y);
            double length1 = Math.sqrt((v1.x * v1.x) + (v1.y * v1.y));
            double length2 = Math.sqrt((v2.x * v2.x) + (v2.y * v2.y));
            double angle = Math.acos(dotproduct / (length1 * length2));
            angle = angle * 180 / Math.PI;
            if (angle < 60) {
                double centerPrev = Math.sqrt(((prev.x - center.x) * (prev.x - center.x)) + ((prev.y - center.y) * (prev.y - center.y)));
                double centerVert = Math.sqrt(((vertex.x - center.x) * (vertex.x - center.x)) + ((vertex.y - center.y) * (vertex.y - center.y)));
                double centerNext = Math.sqrt(((next.x - center.x) * (next.x - center.x)) + ((next.y - center.y) * (next.y - center.y)));
                if (centerPrev < centerVert && centerNext < centerVert) {

                    pointList.add(vertex);
                    //Core.circle(img, vertex, 2, new Scalar(200,0,230));

                    //Core.line(img, vertex, center, new Scalar(0,255,255));
                }
            }
        }

        Point media = new Point();
        media.x = 0;
        media.y = 0;
        int med = 0;
        boolean t = false;
        if (pointList.size() > 0) {
            double dif = Math.sqrt(((pointList.get(0).x - pointList.get(pointList.size() - 1).x) * (pointList.get(0).x - pointList.get(pointList.size() - 1).x)) + ((pointList.get(0).y - pointList.get(pointList.size() - 1).y) * (pointList.get(0).y - pointList.get(pointList.size() - 1).y)));
            if (dif <= 20) {
                t = true;
            }
        }
        for (int i = 0; i < pointList.size() - 1; i++) {

            double d = Math.sqrt(((pointList.get(i).x - pointList.get(i + 1).x) * (pointList.get(i).x - pointList.get(i + 1).x)) + ((pointList.get(i).y - pointList.get(i + 1).y) * (pointList.get(i).y - pointList.get(i + 1).y)));

            if (d > 20 || i + 1 == pointList.size() - 1) {
                Point p = new Point();

                p.x = (int) (media.x / med);
                p.y = (int) (media.y / med);

                //if(p.x>0 && p.x<1024 && p.y<768 && p.y>0){

                processedPoints.add(p);
                //}

                if (t && i + 1 == pointList.size() - 1) {
                    Point ult = new Point();
                    if (processedPoints.size() > 1) {
                        ult.x = (processedPoints.get(0).x + processedPoints.get(processedPoints.size() - 1).x) / 2;
                        ult.y = (processedPoints.get(0).y + processedPoints.get(processedPoints.size() - 1).y) / 2;
                        processedPoints.set(0, ult);
                        processedPoints.remove(processedPoints.size() - 1);
                    }
                }
                med = 0;
                media.x = 0;
                media.y = 0;
            } else {

                media.x = (media.x + pointList.get(i).x);
                media.y = (media.y + pointList.get(i).y);
                med++;


            }
        }


        return processedPoints;
    }

    public void findPalmCenter(Mat img, Point center, Point point, List<Point> points) {

        Core.line(img, new Point(150, 50), new Point(730, 50), new Scalar(255, 0, 0), 2);
        Core.line(img, new Point(150, 380), new Point(730, 380), new Scalar(255, 0, 0), 2);
        Core.line(img, new Point(150, 50), new Point(150, 380), new Scalar(255, 0, 0), 2);
        Core.line(img, new Point(730, 50), new Point(730, 380), new Scalar(255, 0, 0), 2);
        if (points.size() == 1) {
            Core.line(img, center, point, new Scalar(0, 255, 255), 4);
            Core.circle(img, point, 3, new Scalar(255, 0, 255), 3);
            //Core.putText(img, point.toString(), point, Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(0,200,255));

        } else {
            for (int i = 0; i < points.size(); i++) {
                Core.line(img, center, points.get(i), new Scalar(0, 255, 255), 4);
                Core.circle(img, points.get(i), 3, new Scalar(255, 0, 255), 3);
            }
        }
        Core.circle(img, center, 3, new Scalar(0, 0, 255), 3);
        //Core.putText(img, center.toString(), center, Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(0,200,255));

    }

    public void trackMouse(List<Point> points, Point point, Point center, Robot r, boolean on, Mat img, long temp) throws InterruptedException {

        if (on && center.x > 10 && center.y > 10 && point.x > 10 && center.y > 10 && start) {
            current = temp;
            switch (points.size()) {
                case 0:
                    if (act && current - prev > 500) {
                        WAITING_FOR_ACTION_STRING = "Drag & drop";
                        r.mousePress(InputEvent.BUTTON1_DOWN_MASK);
                        act = false;
                    } else {
                        if (current - prev > 500) {
                            Point p = new Point();
                            Point np = new Point();
                            np.x = center.x - last.x;
                            np.y = center.y - last.y;
                            p.x = (int) (-1 * (np.x - 730)) * 1366 / 580;
                            p.y = (int) (np.y - 50) * 768 / 330;
                            if (p.x > 0 && p.x > 0 && p.x < 1367 && p.y < 769) {
                                r.mouseMove((int) p.x, (int) p.y);
                            }

                        }
                    }
                    break;
                case 1:


                    if (act && current - prev > 500) {
                        WAITING_FOR_ACTION_STRING = "Click";
                        r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
                        r.mousePress(InputEvent.BUTTON1_DOWN_MASK);

                        r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

                        r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

                        act = false;
                    } else {
                        if (current - prev > 500) {
                            WAITING_FOR_ACTION_STRING = "Pointer";

                            Point p1 = new Point();
                            p1.x = (int) (-1 * (point.x - 730)) * 1366 / 580;
                            p1.y = (int) (point.y - 50) * 768 / 330;
                            if (p1.x > 0 && p1.x > 0 && p1.x < 1367 && p1.y < 769) {
                                r.mouseMove((int) p1.x, (int) p1.y);
                            }
                            last.x = center.x - point.x;
                            last.y = center.y - point.y;
                        }
                    }
                    break;
                case 2:
                    double angle = calculateAngle(points.get(0), points.get(1), center);
                    r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
                    r.mouseRelease(InputEvent.BUTTON3_DOWN_MASK);
                    if (act && current - prev > 500) {
                        act = false;
                        if ((int) angle < 30) {
                            WAITING_FOR_ACTION_STRING = "Double click";
                            System.out.println("Double click");
                            r.mousePress(InputEvent.BUTTON1_DOWN_MASK);
                            r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
                            r.delay(100);
                            r.mousePress(InputEvent.BUTTON1_DOWN_MASK);
                            r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
                        } else {
                            WAITING_FOR_ACTION_STRING = "Right mouse button";
                            r.mousePress(InputEvent.BUTTON3_DOWN_MASK);
                            r.mouseRelease(InputEvent.BUTTON3_DOWN_MASK);
                        }

                    }
                    break;
                case 3:
                    WAITING_FOR_ACTION_STRING = "Cancel";
                    act = false;
                    break;
                case 4:
                    WAITING_FOR_ACTION_STRING = "Pointer block: waiting for action!";
                    r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

                    prev = temp;
                    act = true;

                    break;

                case 5:
                    WAITING_FOR_ACTION_STRING = "Pointer block: waiting for action!";
                    if (!act) {
                        r.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

                        prev = temp;
                        act = true;
                    }
                    break;
                default:
                    WAITING_FOR_ACTION_STRING = "Waiting for action...";

                    break;
            }

        } else {
            r.mouseRelease(InputEvent.BUTTON1_MASK);
        }
        Core.putText(img, WAITING_FOR_ACTION_STRING, new Point(50, 40), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(200, 0, 0));

    }

    public Point filterAverage(List<Point> buffer, Point point) {
        Point media = new Point();
        media.x = 0;
        media.y = 0;
        for (int i = buffer.size() - 1; i > 0; i--) {
            buffer.set(i, buffer.get(i - 1));
            media.x = media.x + buffer.get(i).x;
            media.y = media.y + buffer.get(i).y;
        }
        buffer.set(0, point);
        media.x = (media.x + buffer.get(0).x) / buffer.size();
        media.y = (media.y + buffer.get(0).y) / buffer.size();
        return media;
    }


    public static void main(String[] args) throws InterruptedException, AWTException {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        HandGestureRecognizer v = new HandGestureRecognizer();
        VideoCapture webcam = new VideoCapture(0);
        webcam.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, 768);
        webcam.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 1024);
        v.setframe(webcam);
        Robot r = new Robot();
        Mat mimm = new Mat();
        Mat mod = new Mat();
        Point center = new Point();
        Point point = new Point();
        List<Point> buffer = new LinkedList<Point>();
        List<Point> pointBuffer = new LinkedList<Point>();
        List<Point> points = new LinkedList<Point>();
        long temp = 0;


        while (!close) {

            if (!webcam.isOpened() && !close) {
                System.out.println("Camera Error");
            } else {
                List<Point> defect = new LinkedList<Point>();
                if (!close) {
                    temp = System.currentTimeMillis();
                    webcam.retrieve(mimm);
                    mod = v.morphFilter(2, 7, v.filterColorHsv(0, 0, 0, 180, 255, 40, mimm));

                    defect = v.defect(mimm, v.searchArea(mimm, mod, false, false, 450), false, 5);

                    if (buffer.size() < 7) {
                        buffer.add(v.centerPalm(mimm, defect));
                    } else {
                        center = v.filterAverage(buffer, v.centerPalm(mimm, defect));
                    }

                    points = v.process(mimm, v.countourList(mod, 200), center);

                    if (points.size() == 1 && pointBuffer.size() < 5) {
                        pointBuffer.add(points.get(0));
                        point = points.get(0);
                    } else {
                        if (points.size() == 1) {
                            point = v.filterAverage(pointBuffer, points.get(0));
                        }
                    }

                    v.findPalmCenter(mimm, center, point, points);


                    v.trackMouse(points, point, center, r, true, mimm, temp);

                    v.frametolabel(mimm);

                }
            }

        }


    }
}