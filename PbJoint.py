from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PbJoint:
    '''
        0   jointIndex          int     the same joint index as the input parameter
        1   jointName           string  the name of the joint, as specified in the URDF (or SDF etc) file
        2   jointType           int     type of the joint, this also implies the number of position and velocity variables.
                                        JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED.
        3   qIndex              int     the first position index in the positional state variables for this body
        4   uIndex              int     the first velocity index in the velocity state variables for this body
        5   flags               int     reserved
        6   jointDamping        float   the joint damping value, as specified in the URDF file
        7   jointFriction       float   the joint friction value, as specified in the URDF file
        8   jointLowerLimit     float   Positional lower limit for slider and revolute (hinge) joints, as specified in the URDF file.
        9   jointUpperLimit     float   Positional upper limit for slider and revolute joints, as specified in the URDF file. Values ignored in case upper limit <lower limit.
        10  jointMaxForce       float   Maximum force specified in URDF (possibly other file formats) Note that this value is not automatically used. You can use maxForce in 'setJointMotorControl2'.
        11  jointMaxVelocity    float   Maximum velocity specified in URDF. Note that the maximum velocity is not used in actual motor control commands at the moment.
        12  linkName            string  the name of the link, as specified in the URDF (or SDF etc.) file
        13  jointAxis           vec3    joint axis in local frame (ignored for JOINT_FIXED)
        14  parentFramePos      vec3    joint position in parent frame
        15  parentFrameOrn      vec4    joint orientation in parent frame (quaternion x,y,z,w)
        16  parentIndex         int     parent link index, -1 for base

        17  objectIndex         int     the index of the belonged object
    '''
    jointIndex: int
    jointName: str
    jointType: int
    qIndex: int
    uIndex: int
    flags: int
    jointDamping: float
    jointFriction: float
    jointLowerLimit: float
    jointUpperLimit: float
    jointMaxForce: float
    jointMaxVelocity: float
    linkName: str
    jointAxis: Tuple[float]
    parentFramePos: Tuple[float]
    parentFrameOrn: Tuple[float]
    parentIndex: int
    objectIndex: int
    jointState: float = 0
    globalAxis: np.ndarray[float] = np.zeros(3)
    globalPos: np.ndarray[float] = np.zeros(3)
    
    def normalize(self, center, scale):
        self.globalPos -= center
        self.globalPos /= scale

@dataclass
class PbJointAxis:
    jointIndex: int
    jointName: str
    jointType: int
    jointLowerLimit: float
    jointUpperLimit: float
    jointState: float
    globalAxis: np.ndarray[float]
    globalPos: np.ndarray[float]

@dataclass
class PbLink:
    bodyUniqueId: int
    linkIndex: int
    linkWorldPosition: np.ndarray[float]
    linkWorldOrientation: np.ndarray[float]
    localInertialFramePosition: np.ndarray[float]
    localInertialFrameOrientation: np.ndarray[float]
    worldLinkFramePosition: np.ndarray[float]
    worldLinkFrameOrientation: np.ndarray[float]

    def normalize(self, center, scale):
        self.linkWorldPosition -= center
        self.linkWorldPosition /= scale
        self.worldLinkFramePosition -= center
        self.worldLinkFramePosition /= scale

