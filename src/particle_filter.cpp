/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.or
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::random_device rd;
	std::default_random_engine gen(rd());
	num_particles = 1;
	is_initialized = true;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		// to test xformation
		p.x = 6.2785;
		p.y = 1.9598;
		p.theta = 0.0;
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine

	std::random_device rd;
	std::default_random_engine gen(rd());
	std::normal_distribution<double> dist_x(0.0, std_pos[0]);
	std::normal_distribution<double> dist_y(0.0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

	if(fabs(yaw_rate) < 0.001) {
		//yaw rate = 0
		for (int i=0; i < num_particles; i++) {
			Particle p = particles[i];
			std::cout << "p.x=" << p.x << " p.y=" << p.y << " p.theta=" << p.theta << " delta_t=" << delta_t << " velocity=" << velocity << " yaw_rate=" << yaw_rate <<std::endl;
			p.x += velocity*delta_t*cos(p.theta)+dist_x(gen)*0;
			p.y += velocity*delta_t*sin(p.theta)+dist_y(gen)*0;
			p.theta += dist_theta(gen)*0;
			while (p.theta>M_PI) p.theta-=2.*M_PI;
    		while (p.theta<-M_PI) p.theta+=2.*M_PI;
			particles[i] = p;
		}

	} else {
		//yaw rate = 0
		for (int i=0; i < num_particles; i++) {
			Particle p = particles[i];
			std::cout << "p.x=" << p.x << " p.y=" << p.y << " p.theta=" << p.theta << " delta_t=" << delta_t << " velocity=" << velocity << " yaw_rate=" << yaw_rate <<std::endl;
			p.x += velocity/yaw_rate*(sin(p.theta+yaw_rate*delta_t) - sin(p.theta))+dist_x(gen)*0;
			p.y += velocity/yaw_rate*(cos(p.theta) - cos(p.theta+yaw_rate*delta_t))+dist_y(gen)*0;
			p.theta += yaw_rate*delta_t + dist_theta(gen)*0;
			while (p.theta>M_PI) p.theta-=2.*M_PI;
    		while (p.theta<-M_PI) p.theta+=2.*M_PI;
			particles[i] = p;
			std::cout << "p.x=" << p.x << " p.y=" << p.y << " p.theta=" << p.theta << std::endl;
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size(); i++) {
		LandmarkObs obs = observations[i];
		double min_dis = std::numeric_limits<double>::max();
		for(int j=0; j<predicted.size(); j++) {
			LandmarkObs m = predicted[j];
			double dis = sqrt((obs.x - m.x) * (obs.x - m.x) + (obs.y - m.y) * (obs.y - m.y));
			if (dis < min_dis) {
				min_dis = dis;
				obs.id = m.id;
			}
		}
		observations[i] = obs;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	//iterate over each particle
	for(int i=0; i<num_particles; i++) {

		Particle p = particles[i];
		//filter the landmarks which are outside sensor_range for this particle
		std::vector<LandmarkObs> landmark_filtered;
		for(int j=0; j<map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
			double dis = sqrt((landmark.x_f - p.x)*(landmark.x_f - p.x)+(landmark.y_f - p.y)*(landmark.y_f - p.y));
			if(dis < sensor_range) {
				LandmarkObs lmark;
				lmark.id = landmark.id_i;
				lmark.x = landmark.x_f;
				lmark.y = landmark.y_f;
				landmark_filtered.push_back(lmark);
			}
		}

		//convert the observations into global coordinates
		std::vector<LandmarkObs> obs_list_global;
		for(int j=0; j<observations.size(); j++) {
			LandmarkObs obs = observations[i];
			LandmarkObs obs_g;
			obs_g.x = obs.x*cos(p.theta)-obs.y*sin(p.theta) + p.x;
			obs_g.y = +obs.x*sin(p.theta)+obs.y*cos(p.theta) + p.y;
			obs_list_global.push_back(obs_g);
		}

		//associate observations with inrange_landmarks
		dataAssociation(landmark_filtered,obs_list_global);

		//update associations in the particle
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		for(int j=0; j<obs_list_global.size(); j++) {
			LandmarkObs obs = obs_list_global[j];
			//std::cout << "obs.id=" << obs.id << std::endl;
			associations.push_back(obs.id);
			sense_x.push_back(obs.x);
			sense_y.push_back(obs.y);
		}
		p = SetAssociations(p, associations, sense_x, sense_y);
		particles[i] = p;

		//Update the weights of particle using a mult-variate Gaussian distribution
		double w = 1.0;
		double sx = std_landmark[0];
		double sy = std_landmark[1];
		for (int j=0; j<p.associations.size(); j++) {
			int lmark_id =  p.associations[j];
			double x = p.sense_x[j];
			double y = p.sense_y[j];
			double mx;
			double my;
			bool found = false;
			for(int k=0; k<landmark_filtered.size(); k++) {
				if (landmark_filtered[k].id == lmark_id) {
					mx = landmark_filtered[k].x;
					my = landmark_filtered[k].y;
					found = true;
					break;
				}
			}
			if (!found) {
				std::cout << "mapping not found. Bug" << std::endl;
				std::exit(1);
			}

			double wj = 1/(2.0*M_PI*sx*sy)* exp(-0.5*((x-mx)*(x-mx)/(sx*sx)
											  +(y-my)*(y-my)/(sy*sy)));
			w *= wj;
		}
		particles[i].weight = w;
		weights[i] = w;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	std::default_random_engine gen(rd());
	std::discrete_distribution<int> d( weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;
	int idx;

	for(int i=0; i < num_particles; i++) {
		idx = d(gen);
		Particle p = particles[idx];
		resampled_particles.push_back(p);
	}

	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

void ParticleFilter::writeBest(Particle best,std::string filename, int time_step) {

	// If first time step create txt file
	if(time_step == 0)
	{
		std::ofstream outfile (filename);
		outfile.close();
	}

	// Append best particle's x,y,theta and associated observations to text file
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);

	dataFile << best.x << " " << best.y << " " << best.theta;
	/*
	for(int i = 0; i < best.sense_x.size(); i++)
	{
		dataFile << " " << best.associations[i] << " " << best.sense_x[i] << " " << best.sense_y[i];
	}
	*/
	dataFile << "\n";

	dataFile.close();
}
