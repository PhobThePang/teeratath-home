#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import speech_recognition as sr

class SpeechCommander:
    def __init__(self):
        rospy.init_node('speech_commander', anonymous=True)

        self.recognizer = sr.Recognizer()

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

        rospy.Subscriber('speech_commands', String, self.speech_callback)

    def speech_callback(self, data):
        command = data.data.lower()
        if "ซ้าย" in command:
            self.turn_left()
        if "ขวา" in command:
            self.turn_right()
        if "ไปข้างหน้า" in command:
            self.forward()

    def turn_left(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.5
        self.cmd_vel_pub.publish(twist_msg)

    def turn_right(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = -0.5
        self.cmd_vel_pub.publish(twist_msg)

    def forward(self):
        twist_msg = Twist()
        twist_msg.linear.x = 2.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)

    def listen(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")

            while not rospy.is_shutdown():
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    command = self.recognizer.recognize_google(audio, language='th-TH')
                    rospy.loginfo("Command: %s", command)

                    self.speech_callback(String(command)) 
                except sr.UnknownValueError:
                    rospy.logwarn("Speech recognition could not understand audio.")
                except sr.RequestError as e:
                    rospy.logerr("Error occurred in speech recognition service: %s", e)

if __name__ == '__main__':
    try:
        commander = SpeechCommander()
        commander.listen()
        
    except rospy.ROSInterruptException:
        pass