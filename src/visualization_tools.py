from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class VisualizationTools:

    @staticmethod
    def plot_line(x, y, publisher, color=(1., 0., 0.), frame="/base_link"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """
        # Construct a line
        line_strip = Marker()
        line_strip.type = Marker.LINE_STRIP
        line_strip.header.frame_id = frame

        # Set the size and color
        line_strip.scale.x = 0.1
        line_strip.scale.y = 0.1
        line_strip.color.a = 1.
        line_strip.color.r = color[0]
        line_strip.color.g = color[1]
        line_strip.color.g = color[2]

        # Fill the line with the desired values
        for xi, yi in zip(x, y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)

    @staticmethod
    def plot_text(text, publisher, size):
        """
        Publishes the text to publisher
        so it can be visualized in rviz as
        text.
        Args:
            text: The text to publish.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            size: the size of the text.
        """
        # Construct a text
        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.header.frame_id = "/base_link"

        # Set the pose, size, and color
        text_marker.pose.position.x = 0.0
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = 0.0
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        text_marker.scale.x = size
        text_marker.scale.y = size
        text_marker.scale.z = size

        text_marker.color.a = 1.
        text_marker.color.r = 1.
        text_marker.color.g = 1.
        text_marker.color.b = 1.

        # Fill the text with the desired values
        text_marker.text = text

        # Publish the text
        publisher.publish(text_marker)
