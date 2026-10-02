using System;
using System.Reflection;
using UnityEngine;

namespace GoiRuntime.PlayerControl
{
    // Saviour.Start IL 000F..00C6 initializes physics before its save/load logic.
    // Controlled startup suppresses Saviour, so reproduce that physical prefix
    // before PlayerControl.Start caches the motors and before cloning the player.
    // Do not call Saviour.Start: it also loads saves and starts native lifecycle work.
    public sealed class NativePhysicsInitialization
    {
        private const BindingFlags Fields = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        private JointMotor2D hingeMotor, sliderMotor, hubMotor;

        private static T Read<T>(Component component, string name)
        {
            var field = component.GetType().GetField(name, Fields);
            if (field == null) throw new InvalidOperationException("Missing native field: " + name);
            object value = field.GetValue(component);
            if (!(value is T)) throw new InvalidOperationException("Invalid native field: " + name);
            return (T)value;
        }

        private static void Write(Component component, string name, JointMotor2D value)
        {
            var field = component.GetType().GetField(name, Fields);
            if (field == null || field.FieldType != typeof(JointMotor2D))
                throw new InvalidOperationException("Missing native motor field: " + name);
            field.SetValue(component, value);
        }

        public static NativePhysicsInitialization ApplyBeforeStart(GameObject player)
        {
            if (player == null) throw new InvalidOperationException("Missing native player");
            var pc = player.GetComponent("PlayerControl");
            if (pc == null) throw new InvalidOperationException("Missing native controller");
            Component saviour = null;
            foreach (var script in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>())
            {
                if (script.GetType().Name != "Saviour" || script.gameObject.scene != player.scene) continue;
                if (Read<Component>(script, "pc") != pc) continue;
                if (saviour != null) throw new InvalidOperationException("Ambiguous native Saviour");
                saviour = script;
            }
            if (saviour == null) throw new InvalidOperationException("Missing native Saviour bindings");
            var hinge = Read<HingeJoint2D>(saviour, "hinge");
            var slider = Read<SliderJoint2D>(saviour, "slider");
            var hub = Read<HingeJoint2D>(saviour, "hubJoint");
            if (hinge != Read<HingeJoint2D>(pc, "hj") || slider != Read<SliderJoint2D>(pc, "sj") ||
                hub != player.GetComponent<HingeJoint2D>() ||
                !hinge.transform.IsChildOf(player.transform) || !slider.transform.IsChildOf(player.transform))
                throw new InvalidOperationException("Unexpected native joint ownership");

            var recipe = new NativePhysicsInitialization();
            recipe.hingeMotor = hinge.motor;
            recipe.hingeMotor.motorSpeed = 0f;
            // Deliberately copy HINGE motor, exactly as Saviour.Start does. Reading
            // slider.motor here would retain the prefab's 4300 instead of native 10000.
            recipe.sliderMotor = hinge.motor;
            recipe.sliderMotor.motorSpeed = 0f;
            recipe.hubMotor = hub.motor;
            recipe.hubMotor.motorSpeed = 0f;
            recipe.hubMotor.maxMotorTorque = 100f; // Native IL constant, not a tuning parameter.

            Write(saviour, "stillHingeMotor", recipe.hingeMotor);
            Write(saviour, "stillSliderMotor", recipe.sliderMotor);
            Write(saviour, "stillHubMotor", recipe.hubMotor);
            Read<Rigidbody2D>(saviour, "cursor").position = Read<Rigidbody2D>(saviour, "hammer").position;
            slider.motor = recipe.sliderMotor;
            hinge.motor = recipe.hingeMotor;
            hub.motor = recipe.hubMotor;
            return recipe;
        }

        public void ValidateAfterStart(GameObject player)
        {
            // Check every clone as well: cached structs are written back on each
            // FixedUpdate, so fixing only the live Joint2D properties is insufficient.
            var pc = player.GetComponent("PlayerControl");
            var hinge = Read<HingeJoint2D>(pc, "hj");
            var slider = Read<SliderJoint2D>(pc, "sj");
            if (!hinge.transform.IsChildOf(player.transform) || !slider.transform.IsChildOf(player.transform))
                throw new InvalidOperationException("Controller references another player's joints");
            Check(hinge.motor, hingeMotor, player.name + ".hinge");
            Check(slider.motor, sliderMotor, player.name + ".slider");
            Check(player.GetComponent<HingeJoint2D>().motor, hubMotor, player.name + ".hub");
            Check(Read<JointMotor2D>(pc, "motor"), hingeMotor, player.name + ".cachedHinge");
            Check(Read<JointMotor2D>(pc, "slider"), sliderMotor, player.name + ".cachedSlider");
        }

        private static void Check(JointMotor2D actual, JointMotor2D expected, string label)
        {
            if (actual.motorSpeed != expected.motorSpeed || actual.maxMotorTorque != expected.maxMotorTorque)
                throw new InvalidOperationException("Native motor initialization mismatch: " + label);
        }
    }
}
