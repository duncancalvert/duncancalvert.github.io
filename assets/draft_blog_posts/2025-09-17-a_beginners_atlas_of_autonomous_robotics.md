---
layout: post
title: The Beginner's Field Guide to Autonomous Robots
date: 2025-09-18 18:04:10
description: An atlas to self-guided machines and mechatronics
tags: AI Machine_Learning Deep_Learning Research Neural_Networks Product_Management Agents Robotics
categories: data-science
typograms: true
---

<br>
<br>
<p style="text-align: center;">
    <em>"We must shape the AI tools that will in turn shape us."</em><br>
    — Reid Hoffman
</p>
<br>
<br>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
      {% include figure.liquid loading="eager" path="assets/img/posts_beginners_guide_to_auto_robots/autonomous_robots_main_photo.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
    Source: Photo by [Kindel Media](https://www.pexels.com/photo/close-up-photo-of-delivert-robots-8566569)
</div>
<br>
<br>

## Table of Contents

---

<!-- TOC -->

- [Table of Contents](#table-of-contents)
- [Robot Form Factors](#robot-form-factors)
  - [Ground Robots](#ground-robots)
  - [Aerial Robots](#aerial-robots)
  - [Aquatic Robots](#aquatic-robots)
  - [Industrial & Service Robots](#industrial--service-robots)
  - [Novel & Bio-Inspired Robots](#novel--bio-inspired-robots)
- [Sensors](#sensors)
  - [Vision and Optical Sensors](#vision-and-optical-sensors)
  - [Motion and Orientation Sensors](#motion-and-orientation-sensors)
  - [Proximity and Distance Sensors](#proximity-and-distance-sensors)
  - [Tactile and Force Sensors](#tactile-and-force-sensors)
  - [Environmental Sensors](#environmental-sensors)
  - [Audio and Vibration Sensors](#audio-and-vibration-sensors)
  - [Biological and Medical Sensors](#biological-and-medical-sensors)
  - [Specialized Application Sensors](#specialized-application-sensors)
- [Actuators: The Muscles of Robots](#actuators-the-muscles-of-robots)
  - [One](#one)
- [Control Systems: The Brain of Robots](#control-systems-the-brain-of-robots)
  - [One](#one)
- [Power Supplies](#power-supplies)
  - [One](#one)
- [Robotics Theory](#robotics-theory)
  - [One](#one)
- [Robotics Ethics & Policy](#robotics-ethics--policy)
  - [One](#one)

<!-- /TOC -->

<br>

## Robot Form Factors

---

<!------------------ Section --------------------->

### Ground Robots

<details>
  <summary><b>Wheeled Robots</b></summary>
  <ul>
    <li><a href="">Autonomous Mobile Robot (AMR)</a>: driverless, self-navigating robots that transport materials and goods independently, using sensors and artificial intelligence (AI) to map their environments, avoid obstacles, and make real-time decisions. Examples include delivery robots, driverless cars, and space rovers.</li>
    <li><a href="https://en.wikipedia.org/wiki/Automated_guided_vehicle">Automated Guided Vehicles (AGV)</a>: portable robots that follows along marked lines, wires on the floor, or uses radio waves, vision cameras, magnets, or lasers for navigation. They are generally used in industrial applications to transport heavy materials around a factory or warehouse.</li>
  </ul>
</details>

<details>
<summary><b>Tracked Robots</b></summary>
  <ul>
    <li><a href="">Continuous Tracks (CT)</a>: robots that use a single, continuous track or multiple continuous tracks for movement</li>
    <li><a href="">Modular Track (MT)</a>: track systems made of individual, connected modules, offering flexibility</li>
    <li><a href="">Omni Track (OT)</a>: track designs that provide omnidirectional movement capabilities, allowing the robot to move in any direction</li>
  </ul>
</details>

<details>
<summary><b>Legged Robots</b></summary>
  <ul>
    <li><a href="">One-legged</a>: pogo stick robots use a hopping motion for navigation</li>
    <li><a href="">Bipeds/Humanoids</a>: two-legged robots (e.g., Atlas, ASIMO)</li>
    <li><a href="">Quadrupeds</a>: dog-like robots (e.g., Boston Dynamics Spot)</li>
    <li><a href="">Hexapods/Multi-legged</a>: insect-inspired robots with six or more legs</li>
  </ul>
</details>

<details>
<summary><b>Specialized and Biomorphic Robots</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Snakebot">Serpentine Robot</a>: snake-like locomotion for confined spaces. Often used for pipe inspection or rescue in tight spaces</li>
    <li><a href="https://en.wikipedia.org/wiki/Ballbot">Ballbot Robots</a>: spherical design for rolling locomotion. Examples include Sphero and the Star Wars-inspired BB-8</li>
  </ul>
</details>

<details>
<summary><b>Hybrid Robots</b></summary>
  <ul>
    <li><a href="">Wheeled Legs</a>: combination of wheels and legs for versatile locomotion</li>
    <li><a href="">Transformable Systems</a>: robots that can change their form factor</li>
  </ul>
</details>

### Aerial Robots

<details>
<summary><b>Multirotors</b></summary>
  <ul>
    <li><a href="">Quadcopters</a>: four-rotor drones</li>
    <li><a href="">Hexacopters</a>: six-rotor aircraft</li>
    <li><a href="">Octocopters</a>: eight-rotor heavy-lift drones</li>
    <li><a href="">Microdrones</a>: miniature flying robots, often inspired by insects.</li>
  </ul>
</details>

<details>
<summary><b>Winged UAVs</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Fixed-wing_aircraft">Fixed-wing UAV</a>: traditional aircraft design</li>
    <li><a href="https://en.wikipedia.org/wiki/Ornithopter">Ornithopter</a>: an aircraft that generates lift and thrust by flapping its wings, mimicking the flight of birds and insects (biomimetic flight)</li>
  </ul>
</details>

<details>
<summary><b>VTOL / eVTOL</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/EVTOL">Electric Vertical Take-Off and Landing (eVTOL)</a>: a variety of vertical take-off and landing (VTOL) aircraft that uses electric power to hover, take off, and land vertically</li>
  </ul>
</details>

<details>
<summary><b>Lighter-Than-Air (LTA)</b></summary>
  <ul>
    <li><a href="">Blimp Drone</a>: uses a lifting gas, like helium, to provide buoyancy and an electric motor for control and propolsion. Requires less energy for lift compared to heavier-than-air drones</li>
    <li><a href="">Balloon Drone</a>: uses a lifting gas, like helium, to provide buoyancy with very limited directional control (carried by the wind). Often used for surveillance or sensing.</li>
  </ul>
</details>

### Aquatic Robots

<details>
<summary><b>Underwater Robots</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Autonomous_underwater_vehicle">Autonomous Underwater Vehicles (AUVs)</a>: submarine-style autonomous systems</li>
    <li><a href="https://en.wikipedia.org/wiki/Remotely_operated_underwater_vehicle">Remotely Operated Underwater Vehicles (ROVs)</a>: tethered submarines controlled remotely</li>
    <li><a href="https://en.wikipedia.org/wiki/Underwater_glider">Underwater Gliders</a>: a variable-buoyancy-driven propulsion robots for long-term missions</li>
    <li><a href="https://en.wikipedia.org/wiki/Robot_fish">Robot Fish</a>: a type of biomimetic robot that has the shape and locomotion of a living fish. Most use body-caudal fin (BCF) propulsion, and can be divided into three categories: single joint (SJ), multi-joint (MJ) and smart material-based "soft-body" design</li>
  </ul>
</details>

<details>
<summary><b>Surface Vessels (USVs)</b></summary>
  <ul>
    <li><a href="">Robotic Boats</a>: unmanned surface vessels. Examples include, autonomous passenger boats, cargo transport, and sensing drones</li>
  </ul>
</details>

### Industrial & Service Robots

<details>
<summary><b>Robotic Arms</b></summary>
  <ul>
    <li><a href="">General Manipulators</a>: 6-DOF industrial arms for manufacturing</li>
    <li><a href="">SCARA Arms</a>: Selective Compliance Assembly Robot Arms for pick-and-place</li>
    <li><a href="">Delta Robots</a>: spider-like parallel robots for high-speed packaging</li>
  </ul>
</details>

<details>
<summary><b>Exoskeletons</b></summary>
  <ul>
    <li><a href="">Wearable Augmentation</a>: human strength and endurance enhancement</li>
    <li><a href="">Rehabilitation Exoskeletons</a>: medical and therapy applications</li>
  </ul>
</details>

<details>
<summary><b>Telepresence Robots</b></summary>
  <ul>
    <li><a href="">Mobile Bases with Screens</a>: remote communication platforms</li>
    <li><a href="">Video Conferencing Robots</a>: mobile telepresence systems</li>
  </ul>
</details>

<details>
<summary><b>Service Robots</b></summary>
  <ul>
    <li><a href="">Kiosk-style Assistants</a>: information and service provision robots</li>
    <li><a href="">Cleaning Bots</a>: autonomous cleaning and maintenance robots</li>
  </ul>
</details>

### Novel & Bio-Inspired Robots

<details>
<summary><b>Soft Robots</b></summary>
  <ul>
    <li><a href="">Flexible Robots</a>: Silicone-based compliant systems</li>
    <li><a href="">Human-safe</a>: Soft materials for safe human-robot interaction</li>
  </ul>
</details>

<details>
<summary><b>Modular / Self-Reconfigurable Robots</b></summary>
  <ul>
    <li><a href="">Self-assembling Swarms</a>: Robots that can reconfigure their structure</li>
    <li><a href="">Modular Systems</a>: Interchangeable robotic components</li>
  </ul>
</details>

<details>
<summary><b>Origami Robots</b></summary>
  <ul>
    <li><a href="">Foldable Robots</a>: Paper-folding inspired designs</li>
    <li><a href="">Deployable Systems</a>: Compact robots that unfold for operation</li>
  </ul>
</details>

<br>

## Sensors

---

<!------------------ Section --------------------->

### Vision and Optical Sensors

<details>
  <summary><b>Camera Systems</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Monocular_vision">Monocular Camera</a>: single camera sensor that captures 2D images for object recognition, navigation, and visual servoing. Widely used due to low cost and computational simplicity.</li>
    <li><a href="https://en.wikipedia.org/wiki/Stereo_camera">Stereo Camera</a>: dual camera system that mimics binocular vision to provide depth perception and 3D spatial understanding through triangulation.</li>
    <li><a href="https://en.wikipedia.org/wiki/Omnidirectional_camera">Omnidirectional Camera</a>: 360-degree field of view cameras using fisheye lenses or multiple camera arrays for complete environmental awareness.</li>
    <li><a href="https://en.wikipedia.org/wiki/Thermal_imaging_camera">Thermal/Infrared Camera</a>: detects heat signatures and temperature variations, useful for night vision, search and rescue, and industrial inspection.</li>
  </ul>
</details>

<details>
<summary><b>Depth and Distance Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Lidar">LiDAR (Light Detection and Ranging)</a>: uses laser pulses to create detailed 3D point clouds of the environment, essential for autonomous navigation and mapping.</li>
    <li><a href="https://en.wikipedia.org/wiki/Structured_light">Structured Light Sensor</a>: projects known light patterns and analyzes deformation to calculate depth and surface geometry.</li>
    <li><a href="https://en.wikipedia.org/wiki/Time-of-flight_camera">Time-of-Flight (ToF) Camera</a>: measures the time light takes to travel to objects and back, providing real-time depth information.</li>
    <li><a href="https://en.wikipedia.org/wiki/Photogrammetry">Photogrammetry Sensors</a>: extract 3D information from multiple 2D photographs taken from different angles.</li>
  </ul>
</details>

<details>
<summary><b>Specialized Optical Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Photodiode">Photodiodes</a>: light-sensitive semiconductors used for simple light detection and optical communication.</li>
    <li><a href="https://en.wikipedia.org/wiki/Optical_flow">Optical Flow Sensors</a>: track visual motion patterns to estimate robot movement and velocity relative to the environment.</li>
    <li><a href="">Hyperspectral Camera</a>: captures images across multiple wavelengths for material identification and chemical analysis.</li>
  </ul>
</details>

### Motion and Orientation Sensors

<details>
<summary><b>Inertial Measurement Units (IMUs)</b></summary>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Accelerometer">Accelerometer</a>: measures linear acceleration in three axes, used for tilt sensing and motion detection.</li>
<li><a href="https://en.wikipedia.org/wiki/Gyroscope">Gyroscope</a>: measures angular velocity and rotational motion for orientation tracking and stabilization.</li>
<li><a href="https://en.wikipedia.org/wiki/Magnetometer">Magnetometer</a>: detects magnetic field strength and direction, commonly used as a digital compass.</li>
<li><a href="https://en.wikipedia.org/wiki/Inertial_measurement_unit">9-DOF IMU</a>: combines accelerometer, gyroscope, and magnetometer for complete orientation sensing.</li>
</ul>
</details>

<details>
<summary><b>Position and Navigation Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Global_Positioning_System">GPS/GNSS</a>: satellite-based positioning system for outdoor localization and navigation.</li>
    <li><a href="https://en.wikipedia.org/wiki/Encoder">Rotary Encoder</a>: measures wheel rotation and motor shaft position for odometry and precise motion control.</li>
    <li><a href="">Linear Encoder</a>: measures linear displacement along a single axis for precise positioning systems.</li>
    <li><a href="https://en.wikipedia.org/wiki/Indoor_positioning_system">Indoor Positioning Systems</a>: beacon-based systems using WiFi, Bluetooth, or UWB for indoor localization.</li>
  </ul>
</details>

### Proximity and Distance Sensors

<details>
<summary><b>Ultrasonic Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Ultrasonic_transducer">Ultrasonic Range Finder</a>: uses sound waves to measure distance to objects, commonly used for obstacle avoidance and parking assistance.</li>
    <li><a href="">Ultrasonic Arrays</a>: multiple ultrasonic sensors arranged to provide wider coverage and improved accuracy.</li>
  </ul>
</details>

<details>
<summary><b>Radar Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Radar">Microwave Radar</a>: uses radio waves to detect objects and measure distance, velocity, and angle, effective in all weather conditions.</li>
    <li><a href="https://en.wikipedia.org/wiki/Automotive_radar">Automotive Radar</a>: specialized radar systems for vehicle collision avoidance and adaptive cruise control.</li>
    <li><a href="">Millimeter Wave Radar</a>: high-frequency radar with excellent resolution for precise object detection.</li>
  </ul>
</details>

<details>
<summary><b>Infrared Proximity Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Passive_infrared_sensor">PIR (Passive Infrared)</a>: detects infrared radiation from warm objects, commonly used for motion detection.</li>
    <li><a href="">Active IR Proximity</a>: emits infrared light and measures reflection for short-range distance measurement.</li>
    <li><a href="">IR Break-beam Sensor</a>: detects when objects interrupt an infrared beam between transmitter and receiver.</li>
  </ul>
</details>

### Tactile and Force Sensors

<details>
<summary><b>Touch and Pressure Sensors</b></summary>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Tactile_sensor">Tactile Sensor Arrays</a>: distributed pressure-sensitive elements that provide detailed touch information for manipulation tasks.</li>
<li><a href="https://en.wikipedia.org/wiki/Pressure_sensor">Pressure Sensors</a>: measure applied force or pressure for grip control and material handling.</li>
<li><a href="">Piezoelectric Sensors</a>: generate electrical signals from mechanical stress, used for vibration and impact detection.</li>
<li><a href="">Capacitive Touch Sensors</a>: detect changes in electrical capacitance caused by touch or proximity.</li>
</ul>
</details>

<details>
<summary><b>Force and Torque Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Load_cell">Load Cell</a>: measures weight and force along specific axes, essential for robotic manipulation and assembly.</li>
    <li><a href="https://en.wikipedia.org/wiki/Torque_sensor">Torque Sensor</a>: measures rotational force applied to joints and actuators for precise control.</li>
    <li><a href="">6-DOF Force/Torque Sensor</a>: measures forces and torques in all six degrees of freedom for complex manipulation tasks.</li>
  </ul>
</details>

### Environmental Sensors

<details>
<summary><b>Chemical and Gas Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Gas_detector">Gas Sensors</a>: detect specific gases for safety monitoring, environmental analysis, and leak detection.</li>
    <li><a href="https://en.wikipedia.org/wiki/pH_meter">pH Sensors</a>: measure acidity/alkalinity of liquids for water quality monitoring and chemical processes.</li>
    <li><a href="">VOC Sensors</a>: detect volatile organic compounds for air quality assessment and industrial safety.</li>
    <li><a href="">Smoke Detectors</a>: specialized sensors for fire detection and safety applications.</li>
  </ul>
</details>

<details>
<summary><b>Climate and Weather Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Thermometer">Temperature Sensors</a>: measure ambient and surface temperatures using thermocouples, RTDs, or thermistors.</li>
    <li><a href="https://en.wikipedia.org/wiki/Hygrometer">Humidity Sensors</a>: measure moisture content in air for environmental monitoring and control.</li>
    <li><a href="https://en.wikipedia.org/wiki/Barometer">Barometric Pressure Sensors</a>: measure atmospheric pressure for weather prediction and altitude estimation.</li>
    <li><a href="https://en.wikipedia.org/wiki/Anemometer">Wind Sensors</a>: measure wind speed and direction for meteorological applications and drone flight control.</li>
  </ul>
</details>

### Audio and Vibration Sensors

<details>
<summary><b>Microphones and Audio Sensors</b></summary>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Microphone">Omnidirectional Microphone</a>: captures sound from all directions for general audio recording and voice recognition.</li>
<li><a href="">Directional Microphone</a>: focuses on sound from specific directions to reduce background noise and improve signal quality.</li>
<li><a href="https://en.wikipedia.org/wiki/Microphone_array">Microphone Arrays</a>: multiple microphones used for sound localization and beamforming.</li>
<li><a href="https://en.wikipedia.org/wiki/Ultrasonic_microphone">Ultrasonic Microphone</a>: detects high-frequency sounds beyond human hearing range.</li>
</ul>
</details>

<details>
<summary><b>Vibration and Acoustic Sensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Accelerometer">Vibration Sensors</a>: specialized accelerometers that detect mechanical vibrations for condition monitoring.</li>
    <li><a href="https://en.wikipedia.org/wiki/Hydrophone">Hydrophones</a>: underwater microphones for marine robotics and sonar applications.</li>
    <li><a href="">Acoustic Emission Sensors</a>: detect stress waves in materials for structural health monitoring.</li>
  </ul>
</details>

### Biological and Medical Sensors

<details>
<summary><b>Biosensors</b></summary>
  <ul>
    <li><a href="https://en.wikipedia.org/wiki/Biosensor">Electrochemical Biosensors</a>: detect biological molecules through electrochemical reactions for medical diagnostics.</li>
    <li><a href="https://en.wikipedia.org/wiki/Pulse_oximetry">Pulse Oximeters</a>: measure blood oxygen saturation and heart rate for health monitoring robots.</li>
    <li><a href="">ECG/EKG Sensors</a>: monitor electrical activity of the heart for medical assistance robots.</li>
    <li><a href="">EMG Sensors</a>: detect muscle electrical activity for prosthetic control and rehabilitation robots.</li>
  </ul>
</details>

### Specialized Application Sensors

<details>
<summary><b>Industrial and Inspection Sensors</b></summary>
<ul>
<li><a href="">Eddy Current Sensors</a>: non-destructive testing sensors that detect flaws in conductive materials.</li>
<li><a href="https://en.wikipedia.org/wiki/X-ray">X-ray Sensors</a>: penetrating radiation sensors for internal inspection and quality control.</li>
<li><a href="">Laser Displacement Sensors</a>: high-precision distance measurement for manufacturing and inspection.</li>
<li><a href="">Current Sensors</a>: monitor electrical current flow in robotic systems for safety and control.</li>
</ul>
</details>

<details>
<summary><b>Multi-modal and Fusion Sensors</b></summary>
  <ul>
    <li><a href="">RGB-D Cameras</a>: combine color and depth information in a single sensor package.</li>
    <li><a href="">Sensor Fusion Systems</a>: integrate multiple sensor types for enhanced perception and reliability.</li>
    <li><a href="">Event-based Cameras</a>: asynchronous vision sensors that respond to changes in brightness rather than capturing frames.</li>
  </ul>
</details>

<br>

## Actuators: The Muscles of Robots

---

<!------------------ Section --------------------->

### One

<br>

## Control Systems: The Brain of Robots

---

<!------------------ Section --------------------->

### One

<br>

## Power Supplies

---

<!------------------ Section --------------------->

### One

<br>

## Robotics Theory

---

<!------------------ Section --------------------->

### One

<br>

## Robotics Ethics & Policy

---

<!------------------ Section --------------------->

### One
