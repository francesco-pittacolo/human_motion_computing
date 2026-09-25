# KSAS-Dataset

This dataset contains information about five movements of the American Kenpo Karate's Blocking Set I captured from 20 volunteers. Each volunter has executed the five first movements of the set with each hand. Additionaly, static noise has been captured from each arm of each volunteer to have instances of absence of movement.

# Structure

The dataset consist in a set of csv files, each one containing a execution of one volunteer. The name of the csv files follows the format a-b-c.csv, where:

- a: number of movement.
- b: id of participant.
- c: 'i' or 'd' indicating if the execution has been made with the left or right arm respectively.

The number of movements are used to identify the movement represented by an instance:

- 0: Absence of movement.
- 1: Upward Block.
- 2: Hammering Inward Block.
- 3: Extended Outward Block.
- 4: Outward Downward Block.
- 5: Rear Elbow Block.

# Data

The total number of instances is 240: 20 volunteers x 2 arms x (5 movements + 1 static noise)

Each instance contains 18 time series, captured using an Android device (XiaoMi Mi A2). The sensors used are compiled in the following table:

| Sensor               | Unit       | Description                                            |
| -------------------- | ---------- | ------------------------------------------------------ |
| Accelerometer        | (m/s²)     | Acceleration force in the axes x, y, and z.            |
| Gravity              | (m/s²)     | Gravity force in the axes x, y, and z.                 |
| Gyroscope            | (rad/s)    | Rate of rotation around axes x, y, and z.              |
| Linear Acceleration  | (m/s²)     | Acceleration without gravity in the axes x, y, and z.  |
| Game Rotation Vector | (unitless) | Rotation vector component along the axes x, y, and z.  |
| Magnetic Field       | (μT)       | Geomagnetic field strength along the axes x, y, and z. |

Each three columns correspond with the measures of that sensor in the three spatial axes.

| Sensor               | Column names                                      |
| -------------------- | ------------------------------------------------- |
| Accelerometer        | accelerometer_x, accelerometer_y, accelerometer_z | 
| Gravity              | gravity_x, gravity_y, gravity_z                   | 
| Gyroscope            | gyros_x, gyros_y, gyros_z                         |
| Linear Acceleration  | lin_accel_x,  lin_accel_y,  lin_accel_z           | 
| Game Rotation Vector | game_rot_vec_x,  game_rot_vec_y,  game_rot_vec_z  | 
| Magnetic Field       | magn_field_x, magn_field_y, magn_field_z          | 


More information about the sensors used can be found in the [Android Sensors Overview](https://developer.android.com/guide/topics/sensors/sensors_overview) page.

# License

Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International

KSAS-Dataset © 2020 by PhyUM - Alberto Casas Ortiz is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
