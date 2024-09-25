.. toctree::
   :maxdepth: 2
   :caption: Contents

High-Level-Design
=================

The PumaGuard system consists of three main components:

1. Central Unit

   This unit contains the inference machine which processes the incoming photos
   from the Trailcam Units. If a Puma was identified, the central unit
   activates lights and sound through one of the Output units.

   .. mermaid::
      :zoom:
      :caption: Flowchart of Central Unit

      flowchart TB
        A("New files from Trailcam Unit?") -- No --> E("Wait") --> A
        A -- Yes --> B("Run ML model")
        B --> C("Was a Puma detected?")
        C -- No --> A
        C -- Yes --> D("Request lights and sound from Output Unit")

2. Trailcam Unit

   This unit contains a trailcam and a small processing unit. The processing
   unit monitors the camera for new photos and sends new photos to the Central
   unit for processing.

   .. mermaid::
      :zoom:
      :caption: Flowchart of Trailcam Unit

      flowchart TB
      A("New files on SD Card?") -- No --> C("Wait") --> A
      A -- Yes --> B("Send new files to Central Unit")

3. Output Unit

   The output unit contains a speaker and lights for scaring the puma away. The
   speaks and lights are controlled by a small processing unit which gets
   instructions from the Central unit.

   .. mermaid::
      :zoom:
      :caption: Flowchart of Output Unit

      flowchart TB
      A("Turn on speakers and lights?") -- No --> C("Wait") --> A
      A -- Yes --> B("Turn on speakers and light")

Below a schematic of the setup. Note that there can be multiple Trailcam and
Output units.

.. mermaid::
   :zoom:
   :caption: High-Level-Design of units

   architecture-beta
       group central(cloud)[Central Unit]

       service pi(server)[Pi] in central
       service ap(internet)[AP] in central

       group camera_1(disk)[Trail Cam]

       service trailcam_1(camera)[Trailcam] in camera_1
       service pizero_camera_1(server)[Pi Zero] in camera_1

       group camera_2(disk)[Trail Cam]

       service trailcam_2(camera)[Trailcam] in camera_2
       service pizero_camera_2(server)[Pi Zero] in camera_2

       group output_1(speakers)[Output]

       service speaker_1(speaker)[Speaker] in output_1
       service lights_1(light)[Lights] in output_1
       service pizero_output_1(server)[Pi Zero] in output_1

       group output_2(speakers)[Output]

       service speaker_2(speaker)[Speaker] in output_2
       service lights_2(light)[Lights] in output_2
       service pizero_output_2(server)[Pi Zero] in output_2

       pizero_camera_1:B -- B:ap
       pizero_camera_2:B -- B:ap
       pizero_output_1:B -- B:ap
       pizero_output_2:B -- B:ap
       pizero_camera_1:B -- B:trailcam_1
       pizero_camera_2:B -- B:trailcam_2
       pizero_output_1:B -- B:speaker_1
       pizero_output_1:B -- B:lights_1
       pizero_output_2:B -- B:speaker_2
       pizero_output_2:B -- B:lights_2
       pi:B -- B:ap
