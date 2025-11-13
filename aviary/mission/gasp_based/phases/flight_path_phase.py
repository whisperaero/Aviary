"""
Generic flight path phase builder for 2DOF trajectory optimization.

This module provides a flexible phase builder that can be configured for various
flight segments using the FlightPathODE.
"""

import numpy as np
from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AlphaModes, SpeedType
from aviary.variable_info.variables import Dynamic


class FlightPathPhaseOptions(AviaryOptionsDictionary):
    """Options class for FlightPathPhase configuration."""

    def declare_options(self):
        # Transcription options
        self.declare(
            name='num_segments',
            types=int,
            default=1,
            desc='The number of segments in transcription creation in Dymos.'
        )

        self.declare(
            name='order',
            types=int,
            default=None,
            desc='The order of polynomials for interpolation in the transcription '
                 'created in Dymos.'
        )

        # Phase configuration options
        self.declare(
            name='ground_roll',
            types=bool,
            default=False,
            desc='True if the aircraft is confined to the ground. Removes altitude rate '
                 'as an output and adjusts the TAS rate equation.'
        )

        self.declare(
            name='clean',
            types=bool,
            default=True,
            desc='If true then no flaps or gear are included. Useful for high-speed '
                 'flight phases.'
        )

        self.declare(
            name='alpha_mode',
            default=AlphaModes.DEFAULT,
            types=AlphaModes,
            desc='How angle of attack is handled: DEFAULT (direct control), '
                 'REQUIRED_LIFT (computed from lift requirement), etc.'
        )

        self.declare(
            name='input_speed_type',
            default=SpeedType.TAS,
            types=SpeedType,
            desc='Whether the speed is given as EAS, TAS, or Mach number'
        )

        # State variable options with sensible defaults
        defaults = {
            'mass_ref': 100_000.0,
            'mass_defect_ref': 100.0,
            'mass_bounds': (1000.0, None),
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        defaults = {
            'distance_ref': 1000.0,
            'distance_bounds': (0.0, None),
        }
        self.add_state_options('distance', units='ft', defaults=defaults)

        defaults = {
            'velocity_ref': 200.0,
            'velocity_bounds': (0.0, 1000.0),
        }
        self.add_state_options('velocity', units='kn', defaults=defaults)

        # Only add altitude state if not ground_roll
        defaults = {
            'altitude_ref': 10000.0,
            'altitude_bounds': (0.0, 50000.0),
            'altitude_constraint_ref': 1000.0,
        }
        self.add_state_options('altitude', units='ft', defaults=defaults)

        # Flight path angle state (not used in ground_roll)
        defaults = {
            'flight_path_angle_ref': np.deg2rad(1),
            'flight_path_angle_defect_ref': 0.01,
            'flight_path_angle_bounds': (-15 * np.pi / 180, 25.0 * np.pi / 180),
        }
        self.add_state_options('flight_path_angle', units='rad', defaults=defaults)

        # Control options
        defaults = {
            'angle_of_attack_ref': np.deg2rad(5),
            'angle_of_attack_bounds': (np.deg2rad(-10), np.deg2rad(25)),
            'angle_of_attack_optimize': True,
        }
        self.add_control_options('angle_of_attack', units='rad', defaults=defaults)

        # Time options
        self.add_time_options(units='s')

        # Mission-specific options
        self.declare(
            'reserve',
            types=bool,
            default=False,
            desc='Designate this phase as a reserve phase and contributes its fuel burn '
                 'towards the reserve mission fuel requirements.'
        )

        self.declare(
            name='target_distance',
            default=None,
            units='m',
            desc='The total distance traveled by the aircraft in this phase.'
        )
        self.declare(
            name='constant_alt',
            default=False,
            types=bool,
            desc='Constant altitude for cruise'
        )

        self.declare(
            name='final_altitude',
            default=None,
            units='ft',
            desc='Target altitude at the end of the phase.'
        )

        self.declare(
            name='final_velocity',
            default=None,
            units='kn',
            desc='Target velocity at the end of the phase.'
        )

        self.declare(
            name='required_available_climb_rate',
            default=None,
            units='ft/min',
            desc='Minimum required climb rate capability throughout the phase.'
        )

        self.declare(
            name='pitch_constraint_bounds',
            default=(0.0, 15.0),
            types=tuple,
            units='deg',
            allow_none=True,
            desc='Tuple containing the lower and upper bounds of the pitch constraint, '
            'with unit string.',
        )

        self.declare(
            name='pitch_constraint_ref',
            default=1.0,
            units='deg',
            desc='Scale factor ref for the pitch constraint.',
        )


class FlightPathPhase(PhaseBuilderBase):
    """
    A generic phase builder for 2DOF flight path segments.

    The phase uses FlightPathODE which implements 2-degree-of-freedom
    equations of motion with states: distance, altitude, velocity, and
    flight path angle.
    """

    default_name = 'flight_path_phase'
    default_ode_class = FlightPathODE
    default_options_class = FlightPathPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Build and configure the flight path phase.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
            Configured Dymos phase object
        """
        # Let parent class create the basic phase
        phase = self.phase = super().build_phase(aviary_options)

        # Get user options
        user_options = self.user_options

        # Extract configuration options
        ground_roll = user_options.get_val('ground_roll')
        alpha_mode = user_options.get_val('alpha_mode')
        final_altitude = user_options.get_val('final_altitude', units='ft')
        final_velocity = user_options.get_val('final_velocity', units='kn')
        required_climb_rate = user_options.get_val('required_available_climb_rate', units='ft/min')
        pitch_constraint_bounds = user_options.get_val('pitch_constraint_bounds', units='deg')
        pitch_constraint_ref = user_options.get_val('pitch_constraint_ref', units='deg')
        constant_alt = user_options.get_val('constant_alt')

        # Configure ODE options
        # self.ode_args['ground_roll'] = ground_roll
        # self.ode_args['clean'] = user_options.get_val('clean')
        # self.ode_args['alpha_mode'] = alpha_mode
        # self.ode_args['input_speed_type'] = user_options.get_val('input_speed_type')

        # Add states
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL
        )

        self.add_state(
            'distance',
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.DISTANCE_RATE
        )

        self.add_state(
            'velocity',
            Dynamic.Mission.VELOCITY,
            Dynamic.Mission.VELOCITY_RATE
        )

        if not ground_roll:
            # Add altitude and flight path angle states only for flight phases
            self.add_state(
                'altitude',
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.ALTITUDE_RATE
            )

            self.add_state(
                'flight_path_angle',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE
            )

        # Add controls based on alpha_mode
        if alpha_mode == AlphaModes.DEFAULT:
            # Direct angle of attack control
            self.add_control('angle_of_attack', Dynamic.Vehicle.ANGLE_OF_ATTACK)

        # Add boundary constraints if specified
        if final_altitude is not None and not ground_roll:
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE,
                loc='final',
                equals=final_altitude,
                units='ft',
                ref=user_options.get_val('altitude_constraint_ref', default=10_000.0)
            )

        if final_velocity is not None:
            phase.add_boundary_constraint(
                Dynamic.Mission.VELOCITY,
                loc='final',
                equals=final_velocity,
                units='kn',
                ref=user_options.get_val('velocity_ref', default=200.0)
            )

        # Add climb rate constraint if specified
        if required_climb_rate is not None and not ground_roll:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                lower=required_climb_rate,
                units='ft/min',
                ref=100.0
            )

        if constant_alt:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                equals=0,
                units='ft/s',
                ref=10.0
            )
        #
        #
        # phase.add_path_constraint(
        #     'fuselage_pitch',
        #     'theta',
        #     lower=pitch_constraint_bounds[0],
        #     upper=pitch_constraint_bounds[1],
        #     units='deg',
        #     ref=pitch_constraint_ref,
        # )

        # Add common timeseries outputs
        self._add_timeseries_outputs(phase, ground_roll)

        return phase

    def _add_timeseries_outputs(self, phase, ground_roll):
        """Add standard timeseries outputs for the phase."""

        # Basic outputs always included
        phase.add_timeseries_output(
            Dynamic.Vehicle.MASS,
            output_name=Dynamic.Vehicle.MASS,
            units='lbm'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.DISTANCE,
            output_name=Dynamic.Mission.DISTANCE,
            units='NM'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY,
            units='kn'
        )

        # phase.add_timeseries_output(
        #     Dynamic.Atmosphere.MACH,
        #     output_name=Dynamic.Atmosphere.MACH,
        #     units='unitless'
        # )
        phase.add_timeseries_output('EAS', units='kn')
        phase.add_timeseries_output(Dynamic.Atmosphere.MACH)

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG,
            output_name=Dynamic.Vehicle.DRAG,
            units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.LIFT,
            output_name=Dynamic.Vehicle.LIFT,
            units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG,
            output_name=Dynamic.Vehicle.DRAG,
            units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            output_name=Dynamic.Vehicle.ANGLE_OF_ATTACK,
            units='deg'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            units='lbm/s'
        )

        # Flight-specific outputs (not for ground roll)
        if not ground_roll:
            phase.add_timeseries_output(
                Dynamic.Mission.ALTITUDE,
                output_name=Dynamic.Mission.ALTITUDE,
                units='ft'
            )

            phase.add_timeseries_output(
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE,
                units='deg'
            )

            phase.add_timeseries_output(
                Dynamic.Mission.ALTITUDE_RATE,
                output_name=Dynamic.Mission.ALTITUDE_RATE,
                units='ft/min'
            )

            phase.add_timeseries_output(
                'load_factor',
                output_name='load_factor',
                units='unitless'
            )
        else:
            # Ground roll specific outputs
            phase.add_timeseries_output(
                'normal_force',
                output_name='normal_force',
                units='lbf'
            )


# Register initial guess metadata
FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='Initial guess for time'
)

FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='Initial guess for mass'
)

FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='Initial guess for distance'
)

FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'),
    desc='Initial guess for velocity'
)

FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='Initial guess for altitude'
)

FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessState('flight_path_angle'),
    desc='Initial guess for flight path angle'
)

FlightPathPhase._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'),
    desc='Initial guess for angle of attack control'
)