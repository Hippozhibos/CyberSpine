"""Tests for the CyberMice"""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import rodent

from CyberMice import Mice
import numpy as np

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.001


def _get_mice_corridor_physics():
  walker = Mice()
  arena = corr_arenas.EmptyCorridor()
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(5, 0, 0),
      walker_spawn_rotation=0,
      physics_timestep=_PHYSICS_TIMESTEP,
      control_timestep=_CONTROL_TIMESTEP)

  env = composer.Environment(
      time_limit=30,
      task=task,
      strip_singleton_obs_buffer_dim=True)

  return walker, env


class MiceTest(parameterized.TestCase):

  def test_can_compile_and_step_simulation(self):
    _, env = _get_mice_corridor_physics()
    physics = env.physics
    for _ in range(100):
      physics.step()

  @parameterized.parameters([
      'egocentric_camera',
      'head',
      'left_arm_root',
      'right_arm_root',
      'root_body',
      'pelvis_body',
  ])
  def test_get_element_property(self, name):
    attribute_value = getattr(Mice(), name)
    self.assertIsInstance(attribute_value, mjcf.Element)

  @parameterized.parameters([
      'actuators',
      'bodies',
      'mocap_tracking_bodies',
      'end_effectors',
      'mocap_joints',
      'observable_joints',
  ])
  def test_get_element_tuple_property(self, name):
    attribute_value = getattr(Mice(), name)
    self.assertNotEmpty(attribute_value)
    for item in attribute_value:
      self.assertIsInstance(item, mjcf.Element)

  def test_set_name(self):
    name = 'fred'
    walker = Mice(name=name)
    self.assertEqual(walker.mjcf_model.model, name)

  @parameterized.parameters(
      'tendons_pos',
      'tendons_vel',
      'actuator_activation',
      'appendages_pos',
      'head_height',
      'sensors_torque',
  )
  def test_evaluate_observable(self, name):
    walker, env = _get_mice_corridor_physics()
    physics = env.physics
    observable = getattr(walker.observables, name)
    observation = observable(physics)
    self.assertIsInstance(observation, (float, np.ndarray))

  def test_proprioception(self):
    walker = Mice()
    for item in walker.observables.proprioception:
      self.assertIsInstance(item, observable_base.Observable)

  def test_can_create_two_mice(self):
    mouse1 = Mice(name='mouse1')
    mouse2 = Mice(name='mouse2')
    arena = corr_arenas.EmptyCorridor()
    arena.add_free_entity(mouse1)
    arena.add_free_entity(mouse2)
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)  # Should not raise an error.

    mouse1.mjcf_model.model = 'mouse3'
    mouse2.mjcf_model.model = 'mouse4'
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)  # Should not raise an error.

if __name__ == '__main__':
  absltest.main()