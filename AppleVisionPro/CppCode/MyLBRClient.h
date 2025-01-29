/**

DISCLAIMER OF WARRANTY

The Software is provided "AS IS" and "WITH ALL FAULTS,"
without warranty of any kind, including without limitation the warranties
of merchantability, fitness for a particular purpose and non-infringement.
KUKA makes no warranty that the Software is free of defects or is suitable
for any particular purpose. In no event shall KUKA be responsible for loss
or damages arising from the installation or use of the Software,
including but not limited to any indirect, punitive, special, incidental
or consequential damages of any character including, without limitation,
damages for loss of goodwill, work stoppage, computer failure or malfunction,
or any and all other commercial damages or losses.
The entire risk to the quality and performance of the Software is not borne by KUKA.
Should the Software prove defective, KUKA is not liable for the entire cost
of any service and repair.


COPYRIGHT

All Rights Reserved
Copyright (C)  2014-2015
KUKA Roboter GmbH
Augsburg, Germany

This material is the exclusive property of KUKA Roboter GmbH and must be returned
to KUKA Roboter GmbH immediately upon request.
This material and the information illustrated or contained herein may not be used,
reproduced, stored in a retrieval system, or transmitted in whole
or in part in any way - electronic, mechanical, photocopying, recording,
or otherwise, without the prior written consent of KUKA Roboter GmbH.


\file
\version {1.9}
*/
#ifndef _KUKA_FRI_MY_LBR_CLIENT_H
#define _KUKA_FRI_MY_LBR_CLIENT_H

#include "friLBRClient.h"
#include "exp_robots.h"

#include <boost/thread.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

namespace py = pybind11;

using namespace KUKA::FRI;

/**
 * \brief Template client implementation.
 */
class MyLBRClient : public LBRClient
{

public:

    /**
    * \brief Constructor.
    */
    MyLBRClient(double freqHz, double amplitude);

    /**
    * \brief Destructor.
    */
    ~MyLBRClient();

    /**
    * \brief Callback for FRI state changes.
    *
    * @param oldState
    * @param newState
    */
    virtual void onStateChange(ESessionState oldState, ESessionState newState);

    /**
    * \brief Callback for the FRI session states 'Monitoring Wait' and 'Monitoring Ready'.
    *
    * If you do not want to change the default behavior, you do not have to implement this method.
    */
    virtual void monitor();

    /**
    * \brief Callback for the FRI session state 'Commanding Wait'.
    *
    * If you do not want to change the default behavior, you do not have to implement this method.
    */
    virtual void waitForCommand();


    /**
    * \brief Callback for the FRI state 'Commanding Active'.
    *
    * If you do not want to change the default behavior, you do not have to implement this method.
    */
    virtual void command();


private:

    // Python Integration
    py::scoped_interpreter guard;

    // Shared Memory and Threading
    boost::mutex dataMutex;
    
    // Function for streaming data in a separate thread
    void runStreamerThread();
    
    // Function to start the Python script for external processing
    void startPythonScript();

    // Pointer to shared memory matrix
    double* matrix;

    // Robot Model
    iiwa14 *myLBR;

    // Joint-related variables (position, velocity, torques)
    double torques[7];     // Torque values for each joint
    double qInitial[7];    // Initial joint positions
    double qApplied[7];    // Commanded joint positions
    double qCurr[7];       // Current measured joint positions
    double qOld[7];        // Previous joint positions
    double tauExternal[7]; // External torques on each joint

    // Time parameters for control loop
    double sampleTime;
    double currentTime;
    double t_pressed;

    // Chosen control point on the robot
    signed int bodyIndex;
    Eigen::Vector3d pointPosition;

    // Joint state variables
    Eigen::VectorXd q;   // Current joint positions
    Eigen::VectorXd dq;  // Joint velocities

    // Command torque vectors (with and without constraints)
    Eigen::VectorXd tau_motion;
    Eigen::VectorXd tau_previous;
    Eigen::VectorXd tau_prev_prev;
    Eigen::VectorXd tau_total;

    // DECLARE VARIABLES FOR YOUR CONTROLLER HERE!!!
    Eigen::MatrixXd M;       // Mass matrix
    Eigen::MatrixXd M_inv;   // Inverse of mass matrix
    Eigen::MatrixXd H;       // Homogeneous transformation matrix
    Eigen::MatrixXd R;       // Rotation matrix
    Eigen::Matrix3d R_z;     // Rotation matrix for z-axis transformations
    Eigen::Matrix3d R_ee_i;  // End-effector rotation matrix
    Eigen::MatrixXd J;       // Jacobian matrix

    // Initial Transformation Matrices
    Eigen::MatrixXd H_rw_ini;  // Initial transformation of external system
    Eigen::MatrixXd p_rw_ini;  // Initial position in external system
    Eigen::MatrixXd H_ini;     // Initial transformation of robot
    Eigen::MatrixXd R_ini;     // Initial rotation of robot
    Eigen::VectorXd p_ini;     // Initial position of robot
    Eigen::VectorXd p;         // Current position
    Eigen::VectorXd p_0_ini;   // Initial reference position
    Eigen::VectorXd p_vp_3d;   // Position difference in 3D space

    // Impedance Control Parameters
    Eigen::MatrixXd Kp;  // Translational stiffness
    Eigen::MatrixXd Kr;  // Rotational stiffness
    Eigen::MatrixXd Bp;  // Translational damping
    Eigen::MatrixXd Br;  // Rotational damping

};

#endif // _KUKA_FRI_MY_LBR_CLIENT_H
