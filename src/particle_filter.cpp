/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Modified on: Jan 25, 2019
 * Author: Sidharth Das
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::numeric_limits;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen; //random engine initialized
  num_particles = 20;  // TODO: Set the number of particles
  
  // create normal distribution in x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int index = 0; index < num_particles; ++index) 
  {
    // Sample from normal distributions.    
    Particle p;
    p.id = index;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0; // particle weights initialized to 1
    particles.push_back (p); // particle appended to vector of particles
    weights.push_back(p.weight);
  }
  is_initialized = true; // particle filter initialized
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen; //random engine initialized
  
  // create normal distribution in x, y, theta for zero-mean noise addition to motion
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (int index = 0; index < num_particles; ++index)
  {
    // avoid dividing by zero
    if (fabs(yaw_rate) < 0.00001)
    {
      // motion model with constant yaw
      particles[index].x = particles[index].x + velocity * cos(particles[index].theta) * delta_t;
      particles[index].y = particles[index].y + velocity * sin(particles[index].theta) * delta_t;
    }
    else
    {
      // motion model with change in yaw
      particles[index].x = particles[index].x + velocity/yaw_rate * (sin(particles[index].theta + yaw_rate * delta_t) - sin(particles[index].theta));
      particles[index].y = particles[index].y + velocity/yaw_rate * (-cos(particles[index].theta + yaw_rate * delta_t) + cos(particles[index].theta));
      particles[index].theta = particles[index].theta + yaw_rate * delta_t;
    }    
    
    // Gaussian noise addition to movement
    particles[index].x = particles[index].x + dist_x(gen);
    particles[index].y = particles[index].y + dist_y(gen);
    particles[index].theta = particles[index].theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int index_obs = 0; index_obs < observations.size(); ++index_obs)
  {
    double min_dist = numeric_limits<double>::max(); // Maximum finite value representable with double type
    int ID_value = -1; // An ID that possibly cannot belong to a map landmark
    for (unsigned int index_pred = 0; index_pred < predicted.size(); ++index_pred)
    {
      double diff = dist(observations[index_obs].x, observations[index_obs].y, predicted[index_pred].x, predicted[index_pred].y);
      if (diff < min_dist)
      {
        min_dist = diff;
        ID_value = predicted[index_pred].id;
      }
      observations[index_obs].id = ID_value;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  double weight_sum = 0.0;
  
  // Loop through each particle while transforming observations to map -co-ordinates, associating data and calculating weight
  for (int index_p = 0; index_p < num_particles; ++index_p)
  {
    double x_p = particles[index_p].x;
    double y_p = particles[index_p].y;
    double theta_p = particles[index_p].theta;
    
    // Create vector for holding trnasformed observations
    vector<LandmarkObs> Transformed_OBS;
    
    // Transform observations from Vehicle co-ordinates to Map co-ordinates
    for (unsigned int index_obs = 0; index_obs < observations.size(); ++index_obs)
    {
      double x_map = observations[index_obs].x * cos(theta_p) - observations[index_obs].y * sin(theta_p) + x_p;
      double y_map = observations[index_obs].x * sin(theta_p) + observations[index_obs].y * cos(theta_p) + y_p;
      Transformed_OBS.push_back (LandmarkObs{observations[index_obs].id, x_map, y_map}); // map co-ordinates appended to vector of observations
    }
    
    // Create vector for holding map landmarks
    vector<LandmarkObs> predicted; 
    
    // Collect map landmarks inside sensor range
    for (unsigned int index_map = 0; index_map < map_landmarks.landmark_list.size(); ++index_map)
    {
      int land_ID = map_landmarks.landmark_list[index_map].id_i;
      float land_x = map_landmarks.landmark_list[index_map].x_f;
      float land_y = map_landmarks.landmark_list[index_map].y_f;
      double pred_distance = dist(x_p, y_p, land_x, land_y);
      if (pred_distance <= sensor_range)
      {
        predicted.push_back(LandmarkObs{land_ID, land_x, land_y});
      }
    }
    
    // Associate transformed observations to collected map landmarks
    dataAssociation(predicted, Transformed_OBS);
    
    // Create vector to append particle associations  
    vector<int> association;
    vector<double> sense_x;
	vector<double> sense_y;
    
    // Reset particle weight to 1
    particles[index_p].weight = 1.0;
    
    // Calculate weight of particle
    double map_x, map_y, mu_x, mu_y, exponent;
    for (unsigned int index_TOBS = 0; index_TOBS < Transformed_OBS.size(); ++index_TOBS)
    {
      map_x =  Transformed_OBS[index_TOBS].x;
      map_y =  Transformed_OBS[index_TOBS].y;
      for (unsigned int index_predicted = 0; index_predicted < predicted.size(); ++index_predicted)
      {
        // Associate prediction with transformed observation
        if (predicted[index_predicted].id == Transformed_OBS[index_TOBS].id)
        {
          mu_x = predicted[index_predicted].x;
          mu_y = predicted[index_predicted].y;
        }
      }
      
      // Multi variate gaussian
      exponent = (pow(map_x - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(map_y - mu_y, 2) / (2 * pow(sig_y, 2)));
      
      // Multiply over observations to obtain particle weight
      particles[index_p].weight = particles[index_p].weight * gauss_norm * exp(-exponent);
      
      // Append particle associations
      association.push_back(Transformed_OBS[index_TOBS].id);
      sense_x.push_back(map_x);
      sense_y.push_back(map_y);
    }
    
    // Sum of weights
    weight_sum = weight_sum + particles[index_p].weight;
    weights[index_p] = particles[index_p].weight;
    
    // Set association for debugging
    SetAssociations(particles[index_p], association, sense_x, sense_y);
    
  }
  
  // Normalize particle weights
  for (int i = 0; i < num_particles; ++i)
  {
    particles[i].weight = particles[i].weight/weight_sum;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Random engine initialized
  std::default_random_engine gen;
  
  // Determine maximum weight of particles
  double weight_max = numeric_limits<double>::min(); // Minimum finite value representable with double type
  for (int index_p = 0; index_p < num_particles; ++index_p)
  {
    if (weights[index_p] > weight_max)
    {
      //weight_max = particles[index_p].weight;
      weight_max = weights[index_p];
    }
  }
  
  // Create uniform distribution to draw uniformly from index of particles
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
  uniform_int_distribution<int> distp(0, num_particles-1);
  
  // Create uniform distribution to draw uniformly between 0 and (2* maximum weight of particles)
  // http://www.cplusplus.com/reference/random/uniform_real_distribution/
  uniform_real_distribution<double> distw(0.0, weight_max);
  
  // Starting index drawn uniformly from index of particles
  int index_particle = distp(gen);
  
  // Increment a variable on the weight representation circle
  double beta = 0.0;
  
  // Create vector of new particles
  vector<Particle> new_particles;
  
  // Weight representation circle
  for (int index_new = 0; index_new < num_particles; ++index_new)
  {
    beta = beta + 2.0 * distw(gen);
    while (beta > weights[index_particle])
    {
      beta = beta - weights[index_particle];
      index_particle = (index_particle + 1) % num_particles;
    }
    
    // Collect resampled particle
    new_particles.push_back(particles[index_particle]);
  }
  
  // Assign resampled particles to particles
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}