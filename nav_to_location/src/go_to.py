#!/usr/bin/env python3

import rospy
from your_package.srv import *

def go_to(location="home"):
    rospy.wait_for_service("/nav/nav_to_location")
    try:
        nav_to = rospy.ServiceProxy("/nav/nav_to_location", NavToLocation)
        print("calling Nav To Location")
        go_home = nav_to(location)
        print("called")
    except rospy.ServiceException as e:
        print("Serivce call failed: %s" % e)

if __name__ == "__main__":
    go_to()