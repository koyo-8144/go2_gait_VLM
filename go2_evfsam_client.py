import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import requests
import numpy as np
import os
import base64

class EvfSamClient(Node):
    def __init__(self):
        super().__init__('evfsam_client_node')
        
        # Create a subscriber to the /camera/color/image_raw topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10  # Queue size
        )
        self.bridge = CvBridge()

        # Server URL
        self.url = "http://0.0.0.0:8000/predict"
        # self.url = "http://10.138.177.122:9000/predict"

        # Text prompt
        self.prompt = "detect a person"

        # Counter for frames
        self.count = 0

        # Directory to save images
        self.save_dir = "image_files"
        os.makedirs(self.save_dir, exist_ok=True)

        self.get_logger().info("RealSense Subscriber Node has started.")

    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the received image
            # cv2.imshow("RealSense Color Image", frame)
            cv2.waitKey(1)  # Add a short delay

            # Convert the color frame to PNG format
            _, buffer = cv2.imencode('.png', frame)

            # Prepare the payload
            files = {
                "text_prompt": (None, self.prompt),
                "image": ("image.png", buffer.tobytes(), "image/png")
            }

            print("send a request")
            # Send the POST request
            response = requests.post(self.url, files=files)

            if response.status_code == 200:
                response_json = response.json()
                self.get_logger().info(f"Processed output received for frame {self.count}")

                # Decode the base64-encoded images
                self.display_and_save_image(response_json["segmentation_image"], "Segmentation Image", "segmentation_image.png")
                self.display_and_save_image(response_json["bounding_box_image"], "Bounding Box Image", "bounding_box_image.png")
                # self.display_and_save_image(response_json["mask_image"], "Mask Image", "mask_image.png")

                # Bounding box coordinates
                xmin = response_json["xmin"]
                ymin = response_json["ymin"]
                xmax = response_json["xmax"]
                ymax = response_json["ymax"]
                self.get_logger().info(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")

            else:
                self.get_logger().error(f"Error: {response.status_code}")
                self.get_logger().error(response.text)

            self.count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def display_and_save_image(self, base64_data, window_name, filename):
        """ Decode, display, and save the image. """
        img_data = base64.b64decode(base64_data)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is not None:
            cv2.imshow(window_name, img)
            # cv2.imwrite(os.path.join(self.save_dir, filename), img)

    def shutdown_callback(self):
        """ Cleanup on shutdown """
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = EvfSamClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RealSense subscriber node.")
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
