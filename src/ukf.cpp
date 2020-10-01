#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);


  //TODO: tune these parameters
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading para// Augmented state dimensionmeter
  lambda_ = 3 - n_aug_;

  NIS_radar_ = 0.0;
  NIS_lidar_ = 0.0;

  //Initialize the covariance matrix P with the identity matrix
  P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
        0, std_laspy_*std_laspy_, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // set weights for the mean and covariance matrix prediction
  // create vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_ /(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for(int i = 1; i < 2 * n_aug_ + 1; ++i) {
      weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

}

UKF::~UKF() {}





void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  //Initialize the state with the first measurement
  if(!is_initialized_) {

    //TODO: need to tune v,yaw and yawd

    if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0,
            0,
            0;
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      //We will not use the velocity measured by the radar because it's speed relative to the ego car and not absolute
      double rho_dot = meas_package.raw_measurements_[2];

      double p_x = rho * cos(phi);
      double p_y = rho * sin(phi);
      x_ << p_x,
            p_y,
            0,
            0,
            0;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //µs -> s
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  
}





void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  /*** 1. UKF augmentation in order to handle the process noise when generating sigma points ***/
  //NB : The process noise nu here is not additive because non-linear. Thus, we add it using the UKF augmentation. 
  // create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;
  /*******************************************************************************************/

  /*** 2. Sigma points generation ***/
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //First column corresponds to the mean
  Xsig_aug.col(0) = x_aug;

  //Each generated column corresponds to the states of a sigma point
  for(int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  /*******************************/

  /*** 3. Sigma points prediction ***/
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    //Extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    //Give the current sigma point to the process model equations 
    double px_p, py_p;

     // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * (sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    } else {
        //Case where the car dives in a straight line
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  /******************************/

  /*** 4. Predict the mean and the covariance matrix ***/
  //Predict state mean
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_pred = x_pred + weights_(i) * Xsig_pred_.col(i);
  }

  //Predict state covariance matrix
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    NormalizeAngle(&x_diff(3));

    P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose();
  }
  /******************************************************/

  x_ = x_pred;
  P_ = P_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  int n_z = 2;

  //Transform sigma points into measurement space using the measurement model (Dim = 2)
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    //Measurement model, the Lidar gives direct info about p_x and p_y
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  //Predict measurement mean
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ +1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //Calculate innovation measurement covariance matrix
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i  = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(&z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //Add measurement noise covariance matrix (Additive here because not non-linear so no need to to UKF augmentation)
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;

  S = S + R;

  //Calculate cross-correlation matrix
  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    //Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    NormalizeAngle(&x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  //Residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //State update
  x_ = x_ + K * z_diff;

  //State covariance matrix update
  P_ = P_ - K * S * K.transpose();


  //Calculate Normalized Innovation Squared for consistency check
  //TODO : Display
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int n_z = 3;

  //Transform sigma points into measurement space using the measurement model (Dim = 3)
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v_x = v*cos(yaw);
    double v_y = v*sin(yaw);

    //Measurement model equations
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                      //rho
    Zsig(1,i) = atan2(p_y,p_x);                               //Phi
    Zsig(2,i) = (p_x*v_x + p_y*v_y)/sqrt(p_x*p_x + p_y*p_y);  //rho_dot
  }

  //Predict measurement mean
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ +1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //Calculate innovation measurement covariance matrix
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i  = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(&z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //Add measurement noise covariance matrix (Additive here because not non-linear so no need to to UKF augmentation)
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  S = S + R;

  //Calculate cross-correlation matrix
  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    //Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //Angle normalization
    NormalizeAngle(&z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    NormalizeAngle(&x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  

  //Calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  //Residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  NormalizeAngle(&z_diff(1));

  //State update
  x_ = x_ + K * z_diff;

  //State covariance matrix update
  P_ = P_ - K * S * K.transpose();


  //Calculate Normalized Innovation Squared for consistency check
  //TODO : Display
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::NormalizeAngle(double* angle) {
  while (*angle> M_PI) *angle-=2.*M_PI;
  while (*angle<-M_PI) *angle+=2.*M_PI;
}