"""CyberMice from Mars"""

import os
import re

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np

_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'CyberMice_CollisionGeom_JointActuated.xml')

_MICE_MOCAP_JOINTS = [
    'RScapula_r1', 'RScapula_r2', 'RScapula_r3', 'RScapula_r4',
    'RShoulder_flexion','RShoulder_adduction', 'RShoulder_rotation', 
    'RElbow_flexion',
    'RRadius_rotation', 'RWrist_adduction', 'RWrist_flexion', 
    'RClavicle_r1', 'RClavicle_r2',
    'LScapula_r1', 'LScapula_r2', 'LScapula_r3', 'LScapula_r4',
    'LShoulder_flexion','LShoulder_adduction', 'LShoulder_rotation', 
    'LElbow_flexion',
    'LRadius_rotation', 'LWrist_adduction', 'LWrist_flexion', 
    'LClavicle_r1', 'LClavicle_r2',
    'RHip_rotation','RHip_flexion','RHip_adduction', 
    'RKnee_flexion', 'RAnkle_flexion', 'RAnkle_rotation',
    'LHip_rotation','LHip_flexion','LHip_adduction', 
    'LKnee_flexion', 'LAnkle_flexion', 'LAnkle_rotation',
    'T_C7_x','C7_C6_y','C6_C5_z',
    'C5_C4_x','C4_C3_y','C3_C2_z',
    'C2_C1_x','C1_head_y','C1_head_z',
    'T_L1_x','L1_L2_y','L2_L3_z',
    'L3_L4_x','L4_L5_y','L5_L6_z',
    'L6_S1_x','L6_S1_y','L6_S1_z',
    ]

_UPRIGHT_POS = (0.0, 0.0, 0.0)
_UPRIGHT_QUAT = (1., 0., 0., 0.)
_TORQUE_THRESHOLD = 60

class Mice(legacy_base.Walker):
    """A muscle-controlled mice with control range scaled to [0.1, 1]."""

    def _build(self,
               params=None,
               name='walker',
               torque_actuators=False,
               physics_timestep: float = 1e-4,
               control_timestep: float = 2e-3,
               initializer=None):
        self.params = params
        self._buffer_size = int(round(control_timestep / physics_timestep))
        root = mjcf.from_path(_XML_PATH)
        self._mjcf_root = root
        if name:
            self._mjcf_root.model = name

        # Remove freejoint.
        root.find('joint', 'free').remove()

        self.body_sites = []
        super()._build(initializer=initializer)

        # modify actuators
        if torque_actuators:
            for actuator in self._mjcf_root.find_all('actuator'):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm


    @property
    def upright_pose(self):
        """Reset pose to upright position."""
        return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)
        

    @property
    def mjcf_model(self):
        """Return the model root."""
        return self._mjcf_root


    @composer.cached_property
    def actuators(self):
        """Return all actuators."""
        return tuple(self._mjcf_root.find_all('actuator'))


    @composer.cached_property
    def root_body(self):
        """Return the body."""
        return self._mjcf_root.find('body', 'Head')

    @composer.cached_property
    def pelvis_body(self):
        """Return the body."""
        return self._mjcf_root.find('body', 'Pelvis')

    @composer.cached_property
    def head(self):
        """Return the head."""
        return self._mjcf_root.find('body', 'Head')

    @composer.cached_property
    def left_arm_root(self):
        """Return the left arm."""
        return self._mjcf_root.find('body', 'LScapula')

    @composer.cached_property
    def right_arm_root(self):
        """Return the right arm."""
        return self._mjcf_root.find('body', 'RScapula')

    @composer.cached_property
    def ground_contact_geoms(self):
        """Return ground contact geoms."""
        return tuple(
            self._mjcf_root.find('body', 'LPedal').find_all('geom') +
            self._mjcf_root.find('body', 'RPedal').find_all('geom') +
            self._mjcf_root.find('body', 'LCarpi').find_all('geom') +
            self._mjcf_root.find('body', 'RCarpi').find_all('geom')
            )

    @composer.cached_property
    def standing_height(self):
        """Return standing height."""
        return self.params['_STAND_HEIGHT']


    @composer.cached_property
    def end_effectors(self):
        """Return end effectors."""
        return (self._mjcf_root.find('body', 'RCarpi'),
                self._mjcf_root.find('body', 'LCarpi'),
                self._mjcf_root.find('body', 'RPedal'),
                self._mjcf_root.find('body', 'LPedal'))

    @composer.cached_property
    def observable_joints(self):
        return tuple(actuator.joint
                 for actuator in self.actuators  #  This lint is mistaken; pylint: disable=not-an-iterable
                 if actuator.joint is not None)

    @composer.cached_property
    def observable_tendons(self):
        return self._mjcf_root.find_all('tendon')

    @composer.cached_property
    def mocap_joints(self):
        return tuple(
            self._mjcf_root.find('joint', name) for name in _MICE_MOCAP_JOINTS)

    @composer.cached_property
    def mocap_joint_order(self):
        return tuple([jnt.name for jnt in self.mocap_joints])  #  This lint is mistaken; pylint: disable=not-an-iterable

    @composer.cached_property
    def bodies(self):
        """Return all bodies."""
        return tuple(self._mjcf_root.find_all('body'))

    @composer.cached_property
    def mocap_tracking_bodies(self):
        """Return bodies for mocap comparison."""
        return tuple(body for body in self._mjcf_root.find_all('body')
                 if not re.match(r'(CyberMice|Carpi|Pedal)', body.name))

    @composer.cached_property
    def primary_joints(self):
        """Return primary (non-vertebra) joints."""
        return tuple(jnt for jnt in self._mjcf_root.find_all('joint')
                    if 'CyberMice' not in jnt.name)

    @composer.cached_property
    def vertebra_joints(self):
        """Return vertebra joints."""
        return tuple(jnt for jnt in self._mjcf_root.find_all('joint')
                 if 'CyberMice' in jnt.name)

    @composer.cached_property
    def primary_joint_order(self):
        joint_names = self.mocap_joint_order
        primary_names = tuple([jnt.name for jnt in self.primary_joints])  # pylint: disable=not-an-iterable
        primary_order = []
        for nm in primary_names:
            primary_order.append(joint_names.index(nm))
        return primary_order

    @composer.cached_property
    def vertebra_joint_order(self):
        joint_names = self.mocap_joint_order
        vertebra_names = tuple([jnt.name for jnt in self.vertebra_joints])  # pylint: disable=not-an-iterable
        vertebra_order = []
        for nm in vertebra_names:
            vertebra_order.append(joint_names.index(nm))
        return vertebra_order

    @composer.cached_property
    def gyro(self):
        """Gyro readings."""
        return self._mjcf_root.find('sensor', 'gyro')
    
    @composer.cached_property
    def accelerometer(self):
        """Accelerometer readings."""
        return self._mjcf_root.find('sensor', 'accelerometer')
    
    @composer.cached_property
    def velocimeter(self):
        """Velocimeter readings."""
        return self._mjcf_root.find('sensor', 'velocimeter')

    @composer.cached_property
    def egocentric_camera(self):
        """Return the egocentric camera."""
        return self._mjcf_root.find('camera', 'egocentric')
        # pass

    @property
    def _xml_path(self):
        """Return the path to th model .xml file."""
        return self.params['_XML_PATH']

    @composer.cached_property
    def joint_actuators(self):
        """Return all joint actuators."""
        return tuple([act for act in self._mjcf_root.find_all('actuator')
                    if act.joint])

    @composer.cached_property
    def joint_actuators_range(self):
        act_joint_range = []
        for act in self.joint_actuators:  #  This lint is mistaken; pylint: disable=not-an-iterable
            associated_joint = self._mjcf_root.find('joint', act.name)
            act_range = associated_joint.dclass.joint.range
            act_joint_range.append(act_range)
        return act_joint_range

    def pose_to_actuation(self, pose):
        # holds for joint actuators, find desired torque = 0
        # u_ref = [2 q_ref - (r_low + r_up) ]/(r_up - r_low)
        r_lower = np.array([ajr[0] for ajr in self.joint_actuators_range])  #  This lint is mistaken; pylint: disable=not-an-iterable
        r_upper = np.array([ajr[1] for ajr in self.joint_actuators_range])  #  This lint is mistaken; pylint: disable=not-an-iterable
        num_tendon_actuators = len(self.actuators) - len(self.joint_actuators)
        tendon_actions = np.zeros(num_tendon_actuators)
        return np.hstack([tendon_actions, (2*pose[self.joint_actuator_order]-
                                        (r_lower+r_upper))/(r_upper-r_lower)])

    @composer.cached_property
    def joint_actuator_order(self):
        joint_names = self.mocap_joint_order
        joint_actuator_names = tuple([act.name for act in self.joint_actuators])  #  This lint is mistaken; pylint: disable=not-an-iterable
        actuator_order = []
        for nm in joint_actuator_names:
            actuator_order.append(joint_names.index(nm))
        return actuator_order

    def _build_observables(self):
        return MiceObservables(self)


class MiceObservables(legacy_base.WalkerObservables):
  """Observables for the Mice."""

  @composer.observable
  def world_zaxis(self):
    """The world's z-vector in this Walker's torso frame."""
    return observable.MJCFFeature('xmat', self._entity.root_body)[6:]
  
  @composer.observable
  def joint_positions(self):
    all_joints = self._entity.mjcf_model.find_all('joint')
    return observable.MJCFFeature('qpos', all_joints)
  
  @composer.observable
  def joint_velocities(self):
    all_joints = self._entity.mjcf_model.find_all('joint')
    return observable.MJCFFeature('qvel', all_joints)

  @composer.observable
  def head_height(self):
    """Observe the head height."""
    return observable.MJCFFeature('xpos', self._entity.head)[2]

  @composer.observable
  def sensors_torque(self):
    """Observe the torque sensors."""
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.sensor.torque,
        corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_THRESHOLD)
        )

  @composer.observable
  def tendons_pos(self):
    return observable.MJCFFeature('length', self._entity.observable_tendons)

  @composer.observable
  def tendons_vel(self):
    return observable.MJCFFeature('velocity', self._entity.observable_tendons)

  @composer.observable
  def actuator_activation(self):
    """Observe the actuator activation."""
    model = self._entity.mjcf_model
    return observable.MJCFFeature('act', model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` with head's position appended."""

    def relative_pos_in_egocentric_frame(physics):
      end_effectors_with_head = (
          self._entity.end_effectors + (self._entity.head,))
      end_effector = physics.bind(end_effectors_with_head).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = \
          np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)

    return observable.Generic(relative_pos_in_egocentric_frame)

  @property
  def proprioception(self):
    """Return proprioceptive information."""
    return [
        self.joints_pos, self.joints_vel,
        self.tendons_pos, self.tendons_vel,
        self.actuator_activation,
        self.body_height, 
        self.end_effectors_pos, 
        self.appendages_pos,
        self.world_zaxis
    ] + self._collect_from_attachments('proprioception')

  @composer.observable
  def gyro(self):
    """Gyro readings."""
    return observable.MJCFFeature('sensordata',
                                    self._entity.mjcf_model.sensor.gyro,
                                    # buffer_size=self._buffer_size,
                                    aggregator='mean')
  
  @composer.observable
  def accelerometer(self):
        """Accelerometer readings."""
        return observable.MJCFFeature(
            'sensordata',
            self._entity.mjcf_model.sensor.accelerometer,
            # buffer_size=self._buffer_size,
            aggregator='mean')
  
  @composer.observable
  def velocimeter(self):
        """Velocimeter readings."""
        return observable.MJCFFeature(
            'sensordata',
            self._entity.mjcf_model.sensor.velocimeter,
            # buffer_size=self._buffer_size,
            aggregator='mean')


  @composer.observable
  def egocentric_camera(self):
    """Observable of the egocentric camera."""
    if not hasattr(self, '_scene_options'):
      # Don't render this walker's geoms.
      self._scene_options = mj_wrapper.MjvOption()
      collision_geom_group = 2
      self._scene_options.geomgroup[collision_geom_group] = 0
      cosmetic_geom_group = 1
      self._scene_options.geomgroup[cosmetic_geom_group] = 0

    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=64, height=64,
                                 scene_option=self._scene_options
                                )
  
  @property
  def vestibular(self):
        """Return vestibular information."""
        return [
            self.gyro, self.accelerometer, self.velocimeter, self.world_zaxis
        ]
