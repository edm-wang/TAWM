#!/usr/bin/env python3
"""
Test script for ManiSkill2 rendering and environment functionality.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set the ManiSkill2 asset directory to use existing assets
os.environ['MS2_ASSET_DIR'] = '/home/romela5090/Admond/TAWM/tawm/envs/ManiSkill2'

# Add the tawm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tawm'))

import hydra
from omegaconf import DictConfig
from tawm.envs import make_env

def test_maniskill_rendering():
    """Test ManiSkill2 rendering and environment functionality."""
    
    # Available ManiSkill2 tasks
    tasks = ['lift-cube', 'pick-cube', 'stack-cube', 'pick-ycb', 'turn-faucet']
    
    print("🎯 Testing ManiSkill2 rendering and environment functionality...")
    print(f"📁 Using assets from: {os.environ['MS2_ASSET_DIR']}")
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Testing task: {task}")
        print(f"{'='*50}")
        
        try:
            # Create a minimal config
            cfg = DictConfig({
                'task': task,
                'obs': 'state',
                'seed': 1,
                'multitask': False
            })
            
            # Create environment
            env = make_env(cfg)
            print(f"✅ Environment created successfully!")
            print(f"   Observation space: {env.observation_space}")
            print(f"   Action space: {env.action_space}")
            print(f"   Max episode steps: {env.max_episode_steps}")
            
            # Test basic environment functionality
            obs = env.reset()
            print(f"   Initial observation shape: {obs.shape}")
            
            # Test rendering
            print(f"   Testing rendering...")
            try:
                render_result = env.render()
                print(f"   ✅ Render successful! Type: {type(render_result)}")
                if hasattr(render_result, 'shape'):
                    print(f"   ✅ Render shape: {render_result.shape}")
                elif isinstance(render_result, dict):
                    print(f"   ✅ Render keys: {list(render_result.keys())}")
                else:
                    print(f"   ✅ Render result: {render_result}")
            except Exception as e:
                print(f"   ❌ Render failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Test a few steps with rendering
            print(f"   Testing steps with rendering...")
            for i in range(3):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                print(f"   Step {i+1}: reward={reward:.3f}, done={done}")
                
                # Test rendering after each step
                try:
                    render_result = env.render()
                    print(f"   ✅ Render after step {i+1} successful!")
                except Exception as e:
                    print(f"   ❌ Render after step {i+1} failed: {e}")
                
                if done:
                    break
            
            # Test timestep control if available
            if hasattr(env, 'step_adaptive_dt'):
                print(f"   ✅ Timestep control available!")
                print(f"   Default dt: {env.default_dt}")
                
                # Test with different timesteps
                test_dts = [0.005, 0.01, 0.02]
                for dt in test_dts:
                    try:
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step_adaptive_dt(action, dt)
                        print(f"   ✅ Timestep {dt}: reward={reward:.3f}")
                        
                        # Test rendering with adaptive timestep
                        try:
                            render_result = env.render()
                            print(f"   ✅ Render with timestep {dt} successful!")
                        except Exception as e:
                            print(f"   ❌ Render with timestep {dt} failed: {e}")
                            
                    except Exception as e:
                        print(f"   ❌ Timestep {dt}: {e}")
            else:
                print(f"   ❌ Timestep control not available")
            
            env.close()
            print(f"✅ Task {task} completed successfully!")
            
        except Exception as e:
            print(f"❌ Task {task} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_maniskill_rendering()
    print("\n" + "="*50)
    print("🎉 ManiSkill2 rendering test completed!")


