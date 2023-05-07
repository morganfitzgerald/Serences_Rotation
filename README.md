# Serences_Rotation
Serences_Rotation

Project outline: 

Investigating the Role of Sensory and Stimulus Noise on Serial Dependence and Memory Biases

Introduction:
Serial dependence is a phenomenon where perception and memory are influenced by prior stimuli or experiences. It has been observed in various sensory modalities, such as vision, audition, and touch. Two questions that remain unanswered are whether serial dependence is a function of sensory noise or stimulus noise, and what factors influence the switch between repulsive and attractive bias. Sensory noise refers to the variability in the sensory input, while stimulus noise refers to the variability in the physical properties of the stimuli themselves.

Hypothesis:
We hypothesize that the strength of serial dependence is modulated by sensory noise rather than stimulus noise. Specifically, we predict that larger biases in memory will be observed when the sensory noise is high, and when a distractor is in the memory delay.

Method:
We will use Holly's data. Participants were presented with stimuli (visual - gradient lines) that vary in their physical orientation. The stimuli will be presented in pairs, with a delay between them (sometimes with a distractor). Participants will be asked to perform a memory task, such as identifying the rotation of the prior stimulus and if it is the same or different from the first one. Sensory noise is manipulated by changing the bandpass filter width, which influences the distribution of energy across frequency bands (closer or farther from white noise). This changes the gradient of the stimulus, and the level of stimulus noise. 

Results:
.....

Conclusion:
This project aims to shed light on the underlying mechanisms of serial dependence and memory biases, and to determine whether they are a function of sensory or stimulus noise. The findings will have important implications for our understanding of how the brain processes information and adapts to noisy environments.

#DATA:
Holly’s Data Columns 
- [x] kappa - bandwidth of the orientation filter for the target stimuli (low kappa means an orientation that looks more noisy and high kappa is an orientation that looks clearer)
- [x] traj - trajectory which is the trajectory of the response (which way they spun the orientation response line)
- [x] orient - probed item orientation (aka correctAngle)
- [x] distractor - non-probed item orientation (aka distractAngle)
- [x] respRT - response time (aka RT) 
- [x] acc - accuracy (the target orientation - the response orientation); aka E
- [x] resp - final orientation response in degrees (aka respAngle)
- [x] subject - participant

#PLAN
filter the data based on kappa 
- Bayesian models would predict that serial dependence is highest in low-high trials, prior would have most influence 
    - still strong for high-high but weaker 
    - smallest for low-low 
- four plots - low undercertainy trials (5,000) and plot serial dependence 
- low uncertainty trails followed by higher 
- low -low 
- high-high
- low-high 
- high - low

- write a loop first and do it the brute force way 
- loop through all trials where 5000, 100 
- start storing those in a data frame 
- arrays or dataframes 
- two variables one current, one previous 
    - if current one is 100, and previous one is 100 sort data here



