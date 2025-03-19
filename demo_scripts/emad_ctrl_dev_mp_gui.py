#!/usr/bin/env python3
from stretch_body.stepper import Stepper
from stretch_body.lift import Lift
import stretch_body.hello_utils as hu
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
import matplotlib.animation as animation
import time

class CtrlDevLift(Lift):
    def __init__(self, usb=None):
        Lift.__init__(self, usb=usb)
        # self.k_c2e = (255/3.3) * 5* 0.15
        self.k_c2e = 69.33/2.166 # ticks / reported

    def move_to(self, x_m, v_m=None, a_m=None, stiffness=None, contact_thresh_pos_N=None,
                contact_thresh_neg_N=None, req_calibration=True, contact_thresh_pos=None,
                contact_thresh_neg=None):
        hu.check_deprecated_contact_model_prismatic_joint(self, 'move_to', contact_thresh_pos_N,
                                                          contact_thresh_neg_N, contact_thresh_pos,
                                                          contact_thresh_neg)
        if req_calibration and not self.motor.status['pos_calibrated']:
            self.logger.warning('%s not calibrated' % self.name.capitalize())
            return
        x_m = min(max(self.soft_motion_limits['current'][0], x_m), self.soft_motion_limits['current'][1])
        print(f"Commanded pos: {x_m}m, limits: {self.soft_motion_limits['current']}")
        stiffness = max(0.0, min(1.0, stiffness)) if stiffness is not None else self.stiffness
        v_r = self.translate_m_to_motor_rad(min(abs(v_m), self.params['motion']['max']['vel_m'])) if v_m is not None else self.vel_r
        a_r = self.translate_m_to_motor_rad(min(abs(a_m), self.params['motion']['max']['accel_m'])) if a_m is not None else self.accel_r
        i_contact_pos, i_contact_neg = self.contact_thresh_to_motor_current(contact_thresh_pos, contact_thresh_neg)
        self.motor.set_command(mode=Stepper.MODE_CTRL_DEV,
                               x_des=self.translate_m_to_motor_rad(x_m),
                               v_des=v_r,
                               a_des=a_r,
                               stiffness=stiffness,
                               i_feedforward=self.i_feedforward,
                               i_contact_pos=i_contact_pos,
                               i_contact_neg=i_contact_neg)

    def stop_effort(self):
        self.motor.set_command(mode=Stepper.MODE_CTRL_DEV,
                               x_des=self.translate_m_to_motor_rad(self.status['pos']),
                               v_des=0.0,
                               a_des=0.0,
                               stiffness=0.0,
                               i_feedforward=0.0,
                               i_contact_pos=0,
                               i_contact_neg=0)
        self.push_command()

    def calc_current_from_effort(self):
        return self.motor.status['effort_ticks'] / self.k_c2e

class LiftMotionGui:
    def __init__(self, start_pos=0.35, stop_pos=1.0, rate='default', stiffness=1.0):
        self.lift = CtrlDevLift()
        if not self.lift.startup(threaded=False):
            raise RuntimeError("Failed to start lift")
        self.lift.motor.disable_sync_mode()
        self.lift.push_command()
        print("Initial motor status:", self.lift.motor.status)

        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.rate = rate
        self.stiffness = stiffness
        self.executing = False
        self.stop_time = None

        # Current run data
        self.time_data = []
        self.effort_data = []
        self.velocity_data = []
        self.reported_current_data = []
        self.calculated_current_data = []
        self.current_diff_data = []
        self.position_data = []
        self.avg_current_diff = 0.0  # New: Running average of differences
        self.diff_count = 0  # New: Number of samples for averaging

        # Previous run data
        self.prev_time_data = []
        self.prev_effort_data = []
        self.prev_velocity_data = []
        self.prev_reported_current_data = []
        self.prev_calculated_current_data = []
        self.prev_current_diff_data = []
        self.prev_position_data = []

        # Figure setup with 6 subplots
        self.fig, self.axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
        self.fig.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9, hspace=0.4)
        self.fig.canvas.set_window_title("CtrlDevLift Motion Profile")

        # Subplot labels
        self.axes[0].set_ylabel('Effort')
        self.axes[1].set_ylabel('Velocity')
        self.axes[2].set_ylabel('Reported Current')
        self.axes[3].set_ylabel('Calculated Current')
        self.axes[4].set_ylabel('Current Diff')
        self.axes[5].set_ylabel('Position')
        self.axes[5].set_xlabel('Time')
        for ax in self.axes:
            ax.grid(True)

        # Current run lines
        self.effort_line, = self.axes[0].plot([], [], label='Effort (Current)', color='blue')
        self.velocity_line, = self.axes[1].plot([], [], label='Velocity (Current)', color='orange')
        self.reported_current_line, = self.axes[2].plot([], [], label='Reported (Current)', color='green')
        self.calculated_current_line, = self.axes[3].plot([], [], label='Calculated (Current)', color='red')
        self.current_diff_line, = self.axes[4].plot([], [], label='Diff (Current)', color='cyan')
        self.position_line, = self.axes[5].plot([], [], label='Position (Current)', color='purple')

        # Previous run lines (dashed)
        self.prev_effort_line, = self.axes[0].plot([], [], label='Effort (Prev)', color='blue', linestyle='--', alpha=0.5)
        self.prev_velocity_line, = self.axes[1].plot([], [], label='Velocity (Prev)', color='orange', linestyle='--', alpha=0.5)
        self.prev_reported_current_line, = self.axes[2].plot([], [], label='Reported (Prev)', color='green', linestyle='--', alpha=0.5)
        self.prev_calculated_current_line, = self.axes[3].plot([], [], label='Calculated (Prev)', color='red', linestyle='--', alpha=0.5)
        self.prev_current_diff_line, = self.axes[4].plot([], [], label='Diff (Prev)', color='cyan', linestyle='--', alpha=0.5)
        self.prev_position_line, = self.axes[5].plot([], [], label='Position (Prev)', color='purple', linestyle='--', alpha=0.5)

        for ax in self.axes:
            ax.legend(loc='upper right')

        # Widgets
        ax_start = plt.axes([0.81, 0.9, 0.15, 0.03])
        ax_stop = plt.axes([0.81, 0.85, 0.15, 0.03])
        ax_rate = plt.axes([0.81, 0.70, 0.15, 0.1])
        ax_stiff = plt.axes([0.81, 0.60, 0.15, 0.03])
        ax_exec = plt.axes([0.81, 0.50, 0.15, 0.03])
        ax_reset = plt.axes([0.81, 0.45, 0.15, 0.03])
        ax_avg_diff = plt.axes([0.81, 0.40, 0.15, 0.03]) 

        self.start_text = TextBox(ax_start, 'Start Pos', initial=str(start_pos))
        self.stop_text = TextBox(ax_stop, 'Stop Pos', initial=str(stop_pos))
        self.rate_radio = RadioButtons(ax_rate, ['default', 'slow', 'fast', 'max'], active=0)
        self.stiff_slider = Slider(ax_stiff, 'Stiffness', 0.0, 1.0, valinit=stiffness)
        self.exec_button = Button(ax_exec, 'Execute', color='limegreen', hovercolor='lightgreen')
        self.reset_button = Button(ax_reset, 'Reset')
        self.avg_diff_text = TextBox(ax_avg_diff, 'avg ratio', initial='N/A', color='lightyellow', hovercolor='white')

        self.start_text.on_submit(self._update_params)
        self.stop_text.on_submit(self._update_params)
        self.rate_radio.on_clicked(self._update_params)
        self.stiff_slider.on_changed(self._update_params)
        self.exec_button.on_clicked(self._execute)
        self.reset_button.on_clicked(self._reset)

        self.anim = animation.FuncAnimation(self.fig, self._animate, interval=10)

    def _update_params(self, val=None):
        try:
            self.start_pos = float(self.start_text.text)
            self.stop_pos = float(self.stop_text.text)
            self.rate = self.rate_radio.value_selected
            self.stiffness = self.stiff_slider.val
        except ValueError:
            print("Invalid input; using last valid values")

    def _execute(self, event):
        if not self.executing:
            self.prev_time_data = self.time_data[:]
            self.prev_effort_data = self.effort_data[:]
            self.prev_velocity_data = self.velocity_data[:]
            self.prev_reported_current_data = self.reported_current_data[:]
            self.prev_calculated_current_data = self.calculated_current_data[:]
            self.prev_current_diff_data = self.current_diff_data[:]
            self.prev_position_data = self.position_data[:]

            self.time_data = []
            self.effort_data = []
            self.velocity_data = []
            self.reported_current_data = []
            self.calculated_current_data = []
            self.current_diff_data = []
            self.position_data = []
            self.avg_current_diff = 0.0  # Reset average
            self.diff_count = 0  # Reset count
            self.start_time = time.time()
            self.stop_time = None

            v_m = self.lift.params['motion'][self.rate]['vel_m']
            a_m = self.lift.params['motion'][self.rate]['accel_m']
            self.lift.move_to(self.start_pos, v_m=v_m, a_m=a_m, stiffness=self.stiffness)
            self.lift.push_command()
            print("After start command:", self.lift.motor.status)
            self.lift.motor.wait_until_at_setpoint(timeout=5.0)
            print("Reached start position, sleeping for 1s")
            time.sleep(1.0)

            self.lift.move_to(self.stop_pos, v_m=v_m, a_m=a_m, stiffness=self.stiffness)
            self.lift.push_command()
            print("After stop command:", self.lift.motor.status)
            self.executing = True

    def _reset(self, event):
        print("Resetting...")
        self.executing = False
        self.stop_time = None
        self.anim.event_source.stop()
        self.lift.stop_effort()
        self.lift.push_command()
        time.sleep(0.5)
        self.lift.pull_status()

        self.time_data = []
        self.effort_data = []
        self.velocity_data = []
        self.reported_current_data = []
        self.calculated_current_data = []
        self.current_diff_data = []
        self.position_data = []
        self.prev_time_data = []
        self.prev_effort_data = []
        self.prev_velocity_data = []
        self.prev_reported_current_data = []
        self.prev_calculated_current_data = []
        self.prev_current_diff_data = []
        self.prev_position_data = []
        self.avg_current_diff = 0.0
        self.diff_count = 0

        self.avg_diff_text.set_val("N/A")
        self._update_plots()
        self.anim = animation.FuncAnimation(self.fig, self._animate, interval=10)
        print("Reset complete, motor status:", self.lift.motor.status)

    def _animate(self, frame):
        if self.executing:
            self.lift.pull_status()
            t = time.time() - self.start_time
            vel = self.lift.motor.status['vel'] if abs(self.lift.motor.status['vel']) > 0.01 else 0.0
            reported_current = self.lift.motor.status['current']
            calculated_current = self.lift.calc_current_from_effort()
            current_diff = reported_current - calculated_current
            self.time_data.append(t)
            self.effort_data.append(self.lift.motor.status['effort_ticks'])
            self.velocity_data.append(vel)
            self.reported_current_data.append(reported_current)
            self.calculated_current_data.append(calculated_current)
            self.current_diff_data.append(current_diff)
            self.position_data.append(self.lift.status['pos'])

            # Update running average of differences
            if reported_current != 0:  # Only include non-zero samples
                self.diff_count += 1
                self.avg_current_diff += (current_diff - self.avg_current_diff) / self.diff_count  # Incremental average
                print(f"t={t:.2f}s | Reported: {reported_current:.3f}A | Calculated: {calculated_current:.3f}A | Diff: {current_diff:.3f} | Avg Diff: {self.avg_current_diff:.3f}")
                self.avg_diff_text.set_val(f"{self.avg_current_diff:.3f}")

            if abs(self.lift.status['pos'] - self.stop_pos) < 0.01 and t > 1.0 and self.stop_time is None:
                self.stop_time = t
                print("Motion stopped at t=%.2fs, logging for 1s more" % t)
                self.lift.stop_effort()
                self.lift.push_command()

            if self.stop_time is not None and t >= self.stop_time + 1.0:
                self.executing = False
                print("Logging complete, motor status:", self.lift.motor.status)

            self._update_plots()

    def _update_plots(self):
        self.effort_line.set_data(self.time_data, self.effort_data)
        self.velocity_line.set_data(self.time_data, self.velocity_data)
        self.reported_current_line.set_data(self.time_data, self.reported_current_data)
        self.calculated_current_line.set_data(self.time_data, self.calculated_current_data)
        self.current_diff_line.set_data(self.time_data, self.current_diff_data)
        self.position_line.set_data(self.time_data, self.position_data)

        self.prev_effort_line.set_data(self.prev_time_data, self.prev_effort_data)
        self.prev_velocity_line.set_data(self.prev_time_data, self.prev_velocity_data)
        self.prev_reported_current_line.set_data(self.prev_time_data, self.prev_reported_current_data)
        self.prev_calculated_current_line.set_data(self.prev_time_data, self.prev_calculated_current_data)
        self.prev_current_diff_line.set_data(self.prev_time_data, self.prev_current_diff_data)
        self.prev_position_line.set_data(self.prev_time_data, self.prev_position_data)

        for ax, data, prev_data in zip(self.axes, [self.effort_data, self.velocity_data, self.reported_current_data,
                                                  self.calculated_current_data, self.current_diff_data, self.position_data],
                                      [self.prev_effort_data, self.prev_velocity_data, self.prev_reported_current_data,
                                       self.prev_calculated_current_data, self.prev_current_diff_data, self.prev_position_data]):
            if data or prev_data:
                ax.relim()
                ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def start(self):
        plt.show()

    def cleanup(self):
        self.anim.event_source.stop()
        self.lift.stop_effort()
        self.lift.shutdown()

if __name__ == "__main__":
    try:
        gui = LiftMotionGui(start_pos=0.35, stop_pos=1.0, rate='default', stiffness=1.0)
        gui.start()
    except Exception as e:
        print(f"GUI failed: {e}")
    finally:
        gui.cleanup()
