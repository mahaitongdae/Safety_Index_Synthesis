from safe_rl import ppo_lagrangian
import gym, safety_gym
import numpy as np
from mujoco_py import *
import time
import xmltodict
from xml.dom import minidom
def example():
	ppo_lagrangian(
		env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
		ac_kwargs = dict(hidden_sizes=(64,64))
	)

def convert(v):
    ''' Convert a value into a string for mujoco XML '''
    if isinstance(v, (int, float, str)):
        return str(v)
    # Numpy arrays and lists
    return ' '.join(str(i) for i in np.asarray(v))

def test_safety_gym():
	env = gym.make('Safexp-CarGoal1-v0')
	# env = gym.make('CrossroadEnd2end-v0',
	# 			   training_task='left')
	act_shape = env.action_space.shape
	print(act_shape)
	a = np.array([0.5,0.5])
	# o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
	env.reset()
	while True:
		o2, r, d, info = env.step(a)
		env.render()
		if d:
			# o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
			env.reset()
		# print(o2)
		print(r)
		print(d)
		print(info)

def test_mujoco():
	# model = load_model_from_path('/home/mahaitong/PycharmProjects/safety-starter-agents/xmls/car.xml')
	xml_path = '/home/mahaitong/PycharmProjects/safety-starter-agents/xmls/car.xml'
	with open(xml_path) as f:
		robot_xml_path = f.read()
	robot_xml = xmltodict.parse(robot_xml_path)

	# Add asset section if missing
	if 'asset' not in robot_xml['mujoco']:
		# old default rgb1: ".4 .5 .6"
		# old default rgb2: "0 0 0"
		# light pink: "1 0.44 .81"
		# light blue: "0.004 0.804 .996"
		# light purple: ".676 .547 .996"
		# med blue: "0.527 0.582 0.906"
		# indigo: "0.293 0 0.508"
		asset = xmltodict.parse('''
	                <asset>
	                    <texture type="skybox" builtin="gradient" rgb1="0.527 0.582 0.906" rgb2="0.1 0.1 0.35"
	                        width="800" height="800" markrgb="1 1 1" mark="random" random="0.001"/>
	                    <texture name="texplane" builtin="checker" height="100" width="100"
	                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
	                    <material name="MatPlane" reflectance="0.1" shininess="0.1" specular="0.1"
	                        texrepeat="10 10" texture="texplane"/>
	                </asset>
	                ''')
		robot_xml['mujoco']['asset'] = asset['asset']

	# Convenience accessor for xml dictionary
	worldbody = robot_xml['mujoco']['worldbody']

	# We need this because xmltodict skips over single-item lists in the tree
	worldbody['body'] = [worldbody['body']]
	if 'geom' in worldbody:
		worldbody['geom'] = [worldbody['geom']]
	else:
		worldbody['geom'] = []
	# Add light to the XML dictionary
	light = xmltodict.parse('''<b>
	            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
	                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
	            </b>''')
	worldbody['light'] = light['b']['light']

	# Add floor to the XML dictionary if missing
	if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
		floor = xmltodict.parse('''
	                <geom name="floor" type="plane" condim="6"/>
	                ''')
		worldbody['geom'].append(floor['geom'])

	# Make sure floor renders the same for every world
	for g in worldbody['geom']:
		if g['@name'] == 'floor':
			g.update({'@size': convert([3.5, 3.5, .1]), '@rgba': '1 1 1 1', '@material': 'MatPlane'})

	# Add cameras to the XML dictionary
	cameras = xmltodict.parse('''<b>
	            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
	            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
	            </b>''')
	worldbody['camera'] = cameras['b']['camera']
	xml_string = xmltodict.unparse(robot_xml)
	doc = minidom.Document()
	n_xml_path = '/home/mahaitong/PycharmProjects/safety-starter-agents/xmls/world.xml'
	with open(n_xml_path, 'w') as f:
		f.write(xml_string)
	f.close()
	model = load_model_from_xml(xml_string)
	sim = MjSim(model)
	sim.reset()
	viewer = MjViewer(sim)
	while True:
		viewer.render()

if __name__ == '__main__':
    test_safety_gym()
