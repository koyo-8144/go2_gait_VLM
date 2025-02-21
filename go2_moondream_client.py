#!/usr/bin/env python3
import argparse
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from sensor_msgs.msg import Image as ROS2Image
import rclpy
from rclpy.node import Node
# from moondream import Moondream, detect_device, LATEST_REVISION
from moondream.hf import detect_device, LATEST_REVISION, Moondream

import cv2
from cv_bridge import CvBridge
import sys 

from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play



class MoonDreamClient(Node):
    def __init__(self, prompt=None):
        super().__init__("moondream_client_node")

        self.url = "http://0.0.0.0:8000/predict"

        self.prompt = "What's going on? Respond with a single sentence."
 
        # Subscribers
        self.image_topic = '/camera/color/image_raw'
        self.image_suber = self.create_subscription(ROS2Image, self.image_topic, self.image_callback, 10)
        self.image_data = None
        self.bridge = CvBridge()  # ROS2 to OpenCV converter


    def image_callback(self, msg):
        # Convert ROS2 image data to a NumPy array
        img_np = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        # Convert NumPy array to PIL image
        self.image_data = Image.fromarray(img_np)
        # print("image ", self.image_data) #<PIL.Image.Image image mode=RGB size=1280x720 at 0x752C9BBBBA00>

        # Convert ROS2 image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Display the image using OpenCV
        cv2.imshow('Robot Camera View', cv_image)
        # Add waitKey to allow image to render and enable keyboard control (exit with 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()
        
        _, buffer = cv2.imencode('.png', cv_image)

        # Prepare the payload
        files = {
            "text_prompt": (None, self.prompt),
            "image": ("image.png", buffer.tobytes(), "image/png")
        }

        # print("Send the POST request")
        # Send the POST request
        response = requests.post(self.url, files=files)

        if response.status_code == 200:
            response_json = response.json()
            # print(f"Processed output received for frame {count}")

            answer = response_json["answer"] 
            print("Answer: ", answer)
            self.speak_text(answer)

        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    
    def speak_text(self, text):
        """Convert text to speech using Google TTS and play it directly."""
        tts = gTTS(text=text, lang="en")
        
        # Save audio to a BytesIO object instead of a file
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        
        # Seek to the beginning of the buffer
        audio_buffer.seek(0)

        # Load and play the audio
        audio = AudioSegment.from_file(audio_buffer, format="mp3")
        play(audio)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=False)
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    moondream_node = MoonDreamClient(prompt=args.prompt)
    rclpy.spin(moondream_node)       # Keeps the node running, processing incoming messages
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()