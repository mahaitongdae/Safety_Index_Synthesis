import tensorflow as tf
from safe_rl.pg.network import *
import safety_gym
import gym

def test_tf1():
    random = tf.get_variable('ran',shape=[1,4],initializer=tf.random_normal_initializer)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(random)
        print(a)

def test_tf1_2():
    env = gym.make('Safexp-CarGoal1-v0')

    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    ac_outs = mlp_actor_critic(x_ph,a_ph, action_space=env.action_space)
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.observation_space.sample()
        feed_dict = {x_ph: obs[np.newaxis]}
        tf.summary.FileWriter('./tmp/', graph=sess.graph)
        val = sess.run(v, feed_dict=feed_dict)
        print(val)


if __name__ == '__main__':
    test_tf1_2()