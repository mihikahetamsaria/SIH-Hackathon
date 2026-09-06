# SIH-Hackathon

*1. Architecture of Project:*
User gives an experiment procedure → AI understands the required sequence → watches the astronaut → recognizes what is happening → compares it with the expected sequence → detects mistakes → tells the astronaut what to do next

*2. Work Division:*

*person 1 : dataset generation*
* segment actions
* create correct sequences
* create incorrect sequences
* create variations
* annotate objects, events and hand-object interactions
* organise dataset

*person 2 : 3D HMR* - optional work, jitna hua will be integrated otherwise nahi
- run pretained HMR models on astroanut in the video
- extract 3d pose info and estimate astronaut orientation
- Provide 3D features for event sequencing

*person 3: 2D Human Pose Modelling*
- person detection
- 2d human pose estimation
- Track astronaut across frames
- Experiment with pose-based features
- Provide 2D pose features for event sequencing

*person 4: Hand–Object Interaction*
- hand detection and tracking
- object detection and tracking
- generate hand-object interaction events 
- Detect touch/grasp/hold/release interactions
- Provide interaction features for event sequencing

*person 5: Event Sequencing & Experiment Validation*
- Define experiment steps/events
- Convert observed actions into events
- Compare observed sequence with expected sequence
- Detection of skipped/ wrong ordered/ incorrect steps
- Determine next expected step
- Generate corrective instructions

*person 6: GUI & Logging*
- Design user interface
- Input experiment and step-by-step procedure
- display current/completed/next steps
- Display errors and corrective instructions
- Generate timestamped event logs
- Store experiment results
- Implement voice/audio alerts
- Display experiment status in real time

person 1: tarannum, person 2 : sanket, person 3 : paanav, person 4 : simran, person 5 : mihika, person 6 : Parth

- sanket work is optional and will be integrated depending on how well it is done
- paanav simran mihika have to work together since their work is dependent on one another 

Dataset options : 
1. https://www.esa.int/esatv/Videos/2017/06/Gripping_experiment_in_space
2. https://www.esa.int/esatv/Videos/2017/07/Grasping_for_space
