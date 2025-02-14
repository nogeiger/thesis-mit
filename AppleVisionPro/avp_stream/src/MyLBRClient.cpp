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

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

#include "MyLBRClient.h"
#include "exp_robots.h"

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <thread>
#include <chrono>

using namespace std;
namespace py = pybind11;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#ifndef NCoef
#define NCoef 1
#endif

static double filterOutput[7][NCoef + 1];
static double filterInput[7][NCoef + 1];


/**
* \brief Initialization
*
*/
MyLBRClient::MyLBRClient(double freqHz, double amplitude)
    :guard{} // Initialize guard (Python interpreter)
{

    /** Initialization */
    // THIS CONFIGURATION MUST BE THE SAME AS FOR THE JAVA APPLICATION!!
    qInitial[0] = -8.87 * M_PI/180;
    qInitial[1] = 60.98 * M_PI/180;
    qInitial[2] = 17.51 * M_PI/180;
    qInitial[3] = -79.85 * M_PI/180;
    qInitial[4] = -24.13 * M_PI/180;
    qInitial[5] = 43.03 * M_PI/180;
    qInitial[6] = 4.14 * M_PI/180;

    // Use Explicit-cpp to create your robot
    myLBR = new iiwa14( 1, "Trey");
    myLBR->init( );

    // Current joint configuration and velocity
    q  = Eigen::VectorXd::Zero( myLBR->nq );
    q_ini = Eigen::VectorXd::Zero( myLBR->nq );
    dq = Eigen::VectorXd::Zero( myLBR->nq );

    // Time variables for control loop
    currentTime = 0;
    sampleTime = 0;

    // Initialize joint torques and joint positions (also needed for waitForCommand()!)
    for( int i=0; i < myLBR->nq; i++ )
    {
        qCurr[i] = qInitial[i];
        qOld[i] = qInitial[i];
        qApplied[i] = 0.0;
        torques[i] = 0.0;
    }

    tau_motion    = Eigen::VectorXd::Zero( myLBR->nq );
    tau_previous  = Eigen::VectorXd::Zero( myLBR->nq );
    tau_prev_prev = Eigen::VectorXd::Zero( myLBR->nq );
    tau_total     = Eigen::VectorXd::Zero( myLBR->nq );

    // ************************************************************
    // INITIALIZE YOUR VECTORS AND MATRICES HERE
    // ************************************************************
    M = Eigen::MatrixXd::Zero( myLBR->nq, myLBR->nq );
    M_inv = Eigen::MatrixXd::Zero( myLBR->nq, myLBR->nq );

    pointPosition[0] = 0.0;
    pointPosition[1] = 0.0;
    pointPosition[2] = 0.085;

    bodyIndex = 7;

    H = Eigen::MatrixXd::Zero( 4, 4 );
    R = Eigen::MatrixXd::Zero( 3, 3 );

    R_z <<  0.0, -1.0,  0.0,
        1.0, 0.0,  0.0,
        0.0, 0.0, 1.0;

    H_ini = Eigen::MatrixXd::Zero( 4, 4 );
    R_ini = Eigen::MatrixXd::Zero( 3, 3 );
    p_ini = Eigen::VectorXd::Zero( 3, 1 );
    p_0_ini = Eigen::VectorXd::Zero( 3, 1 );
    p = Eigen::VectorXd::Zero( 3, 1 );

    J = Eigen::MatrixXd::Zero( 6, myLBR->nq );

    // Translational stiffness
    Kp = Eigen::MatrixXd::Identity( 3, 3 );
    Kp = 400 * Kp;

    // Rotational stiffness
    Kr = Eigen::MatrixXd::Identity( 3, 3 );
    Kr = 150 * Kr;

    // Rotational stiffness
    Kq = Eigen::MatrixXd::Identity( 7, 7 );
    Kq = 10 * Kq;

    // Damping will be calculated at runtime
    // Comment out for constant damping!
    //    Bp = Eigen::MatrixXd::Identity( 3, 3 );
    //    Bp = 40 * Bp;
    //    Br = Eigen::MatrixXd::Identity( 3, 3 );
    //    Br = 10 * Br;

    // ************************************************************
    // AVP streamer
    // ************************************************************

    // Unock mutex initially
    dataMutex.unlock();
    boost::thread(&MyLBRClient::runStreamerThread, this).detach();

    // Start the Python script
    startPythonScript();

    // Transformation matrices of AVP
    H_avp_rw_ini = Eigen::MatrixXd::Identity( 4, 4 );
    R_avp_rw_ini = Eigen::MatrixXd::Identity( 3, 3 );
    p_avp_rw_ini = Eigen::VectorXd::Zero( 3 );

    p_avp_rw = Eigen::VectorXd::Zero( 3 );
    p_avp_rw_prev = Eigen::VectorXd::Zero( 3 );
    p_avp_rw_prev_prev = Eigen::VectorXd::Zero( 3 );

    R_avp_rw = Eigen::MatrixXd::Identity( 3, 3 );
    R_avp_rw_prev = Eigen::MatrixXd::Identity( 3, 3 );
    R_avp_rw_prev_prev = Eigen::MatrixXd::Identity( 3, 3 );

    // # definitions can be found here: https://github.com/Improbable-AI/VisionProTeleop
    matrix_rw = new double[16];                     // wrist

    // ************************************************************
    // Store data
    // ************************************************************

    // Open a single binary file
    File_data.open("/home/newman_lab/Desktop/noah_repo/thesis-mit/AppleVisionPro/avp_stream/prints/File_data.bin", std::ios::binary);
    if (!File_data) {
        std::cerr << "Error opening file for writing!" << std::endl;
    }

    // ************************************************************
    // INCLUDE FT-SENSOR
    // ************************************************************
    // Weight: 0.2kg (plate) + 0.255kg (sensor) = 0.455kg

    f_ext_ee = Eigen::VectorXd::Zero( 3 );
    m_ext_ee = Eigen::VectorXd::Zero( 3 );
    f_ext_0 = Eigen::VectorXd::Zero( 3 );
    m_ext_0 = Eigen::VectorXd::Zero( 3 );
    F_ext_0 = Eigen::VectorXd::Zero( 6 );

    AtiForceTorqueSensor ftSensor("172.31.1.1");
    
    // Start threading for force sensor
    mutexFTS.unlock();
    boost::thread(&MyLBRClient::forceSensorThread, this).detach();

    printf( "Sensor Activated. \n\n" );

    // ************************************************************
    // Initial print
    // ************************************************************

    printf( "Exp[licit](c)-cpp-FRI, https://explicit-robotics.github.io \n\n" );
    printf( "Robot '" );
    printf( "%s", myLBR->Name );
    printf( "' initialised. Ready to rumble! \n\n" );

}


/**
* \brief Destructor
*
*/
MyLBRClient::~MyLBRClient()
{

    boost::interprocess::shared_memory_object::remove("SharedMemory_AVP");
    delete myLBR;
    delete this->ftSensor;

    //    if (File_data.is_open()) {
    //        File_data.close();
    //    }

}


/**
* \brief Implements an IIR Filter which is used to send the previous joint position to the command function, so that KUKA's internal friction compensation can be activated. The filter function was generated by the application WinFilter (http://www.winfilter.20m.com/).
*
* @param NewSample The current joint position to be provided as input to the filter.
*/
void iir(double NewSample[7])
{
    double ACoef[ NCoef+1 ] = {
        0.05921059165970496400,
        0.05921059165970496400
    };

    double BCoef[ NCoef+1 ] = {
        1.00000000000000000000,
        -0.88161859236318907000
    };

    int n;

    // Shift the old samples
    for ( int i=0; i<7; i++ )
    {
        for( n=NCoef; n>0; n-- )
        {
            filterInput[i][n] = filterInput[i][n-1];
            filterOutput[i][n] = filterOutput[i][n-1];
        }
    }

    // Calculate the new output
    for (int i=0; i<7; i++)
    {
        filterInput[i][0] = NewSample[i];
        filterOutput[i][0] = ACoef[0] * filterInput[i][0];
    }

    for (int i=0; i<7; i++)
    {
        for(n=1; n<=NCoef; n++)
            filterOutput[i][0] += ACoef[n] * filterInput[i][n] - BCoef[n] * filterOutput[i][n];
    }
}

//******************************************************************************
void MyLBRClient::onStateChange(ESessionState oldState, ESessionState newState)
{
    LBRClient::onStateChange(oldState, newState);
    // react on state change events
    switch (newState)
    {
    case MONITORING_WAIT:
    {
        break;
    }
    case MONITORING_READY:
    {
        sampleTime = robotState().getSampleTime();
        break;
    }
    case COMMANDING_WAIT:
    {
        break;
    }
    case COMMANDING_ACTIVE:
    {
        break;
    }
    default:
    {
        break;
    }
    }
}

//******************************************************************************
void MyLBRClient::monitor()
{

    // Copied from FRIClient.cpp
    robotCommand().setJointPosition(robotState().getCommandedJointPosition());

    // Copy measured joint positions (radians) to _qcurr, which is a double

    memcpy( qCurr, robotState().getMeasuredJointPosition(), 7*sizeof(double) );

    // Initialise the q for the previous NCoef timesteps

    for( int i=0; i<NCoef+1; i++ )
    {
        iir(qCurr);
    }
}

//******************************************************************************
void MyLBRClient::waitForCommand()
{
    // If we want to command torques, we have to command them all the time; even in
    // waitForCommand(). This has to be done due to consistency checks. In this state it is
    // only necessary, that some torque vlaues are sent. The LBR does not take the
    // specific value into account.

    if(robotState().getClientCommandMode() == TORQUE){

        robotCommand().setTorque(torques);
        robotCommand().setJointPosition(robotState().getIpoJointPosition());            // Just overlaying same position
    }

}

//******************************************************************************
void MyLBRClient::command()
{

    // ************************************************************
    // Read out relative potisions in AVP coordinates

    if(currentTime < sampleTime)
    {
        startPythonScript();
    }

    // Lock mutex and update local variables from shared memory
    double* h_rw;

    dataMutex.lock();

    h_rw = matrix_rw;

    dataMutex.unlock();

    // Convert APV transformation to Eigen
    Eigen::MatrixXd H_avp_rw = Eigen::Map<Eigen::MatrixXd>(h_rw, 4, 4);         

    // Rotation of knuckle with respect to avp
    R_avp_rw = H_avp_rw.transpose().block< 3, 3 >( 0, 0 );

    Eigen::MatrixXd R_corrected = R_avp_rw;
    R_corrected.col(0) = R_avp_rw.col(0);        // X remains the same
    R_corrected.col(1) = -R_avp_rw.col(2);       // Z becomes Y (inverted)
    R_corrected.col(2) = R_avp_rw.col(1);        // Y becomes Z

    R_avp_rw = R_corrected;

    // A simple filter for the rotation
    if( currentTime < sampleTime )
    {
        R_avp_rw_prev = R_avp_rw;
        R_avp_rw_prev_prev = R_avp_rw;
    }
    R_avp_rw = ( R_avp_rw + R_avp_rw_prev + R_avp_rw_prev_prev ) / 3;

    // Positon of knuckle with respect to avp
    p_avp_rw = H_avp_rw.transpose().block< 3, 1 >( 0, 3 );

    // A simple filter for the translation
    if( currentTime < sampleTime )
    {
        p_avp_rw_prev = p_avp_rw;
        p_avp_rw_prev_prev = p_avp_rw;
    }
    p_avp_rw = ( p_avp_rw + p_avp_rw_prev + p_avp_rw_prev_prev ) / 3;


    // ****************************************************matrix********
    // Get FTSensor data
    double* fts_bt;

    mutexFTS.lock();

    fts_bt = f_sens_ee;

    mutexFTS.unlock();

    f_ext_ee[0] = fts_bt[0];
    f_ext_ee[1] = fts_bt[1];
    f_ext_ee[2] = fts_bt[2];
    m_ext_ee[0] = fts_bt[3];
    m_ext_ee[1] = fts_bt[4];
    m_ext_ee[2] = fts_bt[5];

    // Convert to robot base coordinates
    f_ext_0 = R * f_ext_ee;
    m_ext_0 = R * m_ext_ee;


    // ************************************************************
    // Get robot measurements

    memcpy( qOld, qCurr, 7*sizeof(double) );
    memcpy( qCurr, robotState().getMeasuredJointPosition(), 7*sizeof(double) );
    memcpy( tauExternal, robotState().getExternalTorque(), 7*sizeof(double) );

    for (int i=0; i < myLBR->nq; i++)
    {
        q[i] = qCurr[i];
    }

    for (int i=0; i < 7; i++)
    {
        dq[i] = (qCurr[i] - qOld[i]) / sampleTime;
    }

    // ************************************************************
    // Calculate kinematics and dynamics

    // Transformation and Rotation Matrix
    //    H = myLBR->getForwardKinematics( q, bodyIndex, pointPosition );
    H = myLBR->getForwardKinematics( q );
    R = H.block< 3, 3 >( 0, 0 );
    p = H.block< 3, 1 >( 0, 3 );

    //  Get initial transfomation of first iteration
    if(currentTime < sampleTime)
    {
        H_ini = H;
        R_ini = R;
        p_ini = p;

        // Get initial AVP transformation
        p_avp_rw_ini = p_avp_rw;
        R_avp_rw_ini = R_avp_rw;
    }

    // Jacobian, translational and rotation part
    //J = myLBR->getHybridJacobian( q, pointPosition );
    J = myLBR->getHybridJacobian( q );
    Eigen::MatrixXd J_v = J.block(0, 0, 3, 7);
    Eigen::MatrixXd J_w = J.block(3, 0, 3, 7);

    // Mass matrix
    // Adapt mass matrix to prevent high accelerations at last joint
    M = myLBR->getMassMatrix( q );
    M( 6, 6 ) = 40 * M( 6, 6 );
    Eigen::MatrixXd M_inv = M.inverse();

    // Cartesian mass matrix
    double k = 0.01;
    Eigen::MatrixXd Lambda = getLambdaLeastSquares(M, J, k);
    Eigen::MatrixXd Lambda_v = getLambdaLeastSquares(M, J_v, k);
    Eigen::MatrixXd Lambda_w = getLambdaLeastSquares(M, J_w, k);

    // ****************** Convert AVP displacement to robot coordinates ******************

    // Displacement from initial position
    Eigen::VectorXd del_p_avp_rw = p_avp_rw - p_avp_rw_ini;

    // Transform to homogeneous coordinates
    Eigen::VectorXd del_p_avp_rw_4d = Eigen::VectorXd::Zero(4, 1);
    del_p_avp_rw_4d[3] = 1;
    del_p_avp_rw_4d.head<3>() = del_p_avp_rw;

    // Transformation to robot base coordinates
    Eigen::MatrixXd H_0_avp = Eigen::MatrixXd::Zero( 4, 4 );
    H_0_avp.block<3, 3>(0, 0) = R_z;
    H_0_avp.block<3, 1>(0, 3) = p_ini;

    // Extract 3x1 position
    Eigen::VectorXd p_0_4d = H_0_avp * del_p_avp_rw_4d;
    Eigen::VectorXd p_0 = p_0_4d.block<3, 1>(0, 0);


    // ****************** Convert AVP rotation to robot coordinates ******************

    // Comment out to keep initial robot configuration
    //Eigen::Matrix3d R_ee_des = R.transpose() * R_ini;

    // Change rotation based on Apple Vision Pro
    Eigen::Matrix3d del_R = R_avp_rw_ini.transpose() * R_avp_rw;                // Absolute rotation of apple vision pro

    Eigen::Matrix3d R_ee_des =  R.transpose() * R_ini * del_R;

    // Transform rotations to quaternions
    Eigen::Quaterniond Q(R_ee_des);
    Q.normalize();

    // Extract rotation angle
    double theta = 2 * acos( Q.w() );
    double eps = 0.01;
    if( theta <  0.01 ){
        theta = theta + eps;
    }

    // Scale desired angle
    int scaleFact = 1;
    theta = theta / scaleFact;

    // Compute norm factor, handle edge case for small theta
    double sin_half_theta = sin(theta / 2);
    double norm_fact;
    if (fabs(sin_half_theta) < 1e-6) {  // Handle small-angle case
        norm_fact = 1.0;  // Default to 1, or handle separately
    } else {
        norm_fact = 1.0 / sin_half_theta;
    }

    Eigen::VectorXd u_ee = Eigen::VectorXd::Zero( 3, 1 );
    u_ee[0] = norm_fact * Q.x();
    u_ee[1] = norm_fact * Q.y();
    u_ee[2] = norm_fact * Q.z();

    // Transform to robot base coordinates
    Eigen::VectorXd u_0 = R * u_ee;

    // ************************************************************
    // Translational task-space impedance controller

    Eigen::VectorXd dx = J_v * dq;
    Eigen::VectorXd del_p = (p_0 - p);

    // Damping design
    double damping_factor_v = 0.7;
    Eigen::Vector3d Kp_diag = Kp.diagonal();
    double alpha_v = compute_alpha(Lambda_v, Kp_diag, damping_factor_v);
    Eigen::MatrixXd Bp = alpha_v * Kp;

    // Calculate force
    Eigen::VectorXd f = Kp * del_p - Bp * dx;

    // Convert to torques
    Eigen::VectorXd tau_translation = J_v.transpose() * f;


    // ************************************************************
    // Rotational task-space impedance controler
    Eigen::VectorXd omega = J_w * dq;

    // Damping design
    double damping_factor_r = 0.7;
    Eigen::Vector3d Kr_diag = Kr.diagonal();
    double alpha_w = compute_alpha(Lambda_w, Kr_diag, damping_factor_r);
    Eigen::MatrixXd Br = alpha_w * Kr;

    // Calculate moment
    Eigen::VectorXd m = Kr * u_0 * theta - Br * omega;

    // Convert to torques
    Eigen::VectorXd tau_rotation = J_w.transpose() * m;


    // ************************************************************
    // Nullspace joint space stiffness
    Eigen::VectorXd J_bar = M_inv * J.transpose() * Lambda;
    Eigen::VectorXd N = Eigen::MatrixXd::Identity(7, 7) - J.transpose() * J_bar.transpose();

    Eigen::VectorXd tau_kq = Kq * (q_ini - q);

    // ************************************************************
    // Control torque
    tau_motion = tau_translation + tau_rotation + (N * tau_kq);

    // Comment out for only gravity compensation
    //    tau_motion = Eigen::VectorXd::Zero( myLBR->nq );

    // Include joint limits
    tau_motion = myLBR->addIIWALimits( q, dq, M, tau_motion, 0.004 );


    // ************************************************************
    // Write data in a file 

    // Buffer binary data
    //    buffer.write(reinterpret_cast<const char*>(&currentTime), sizeof(currentTime));
    //    buffer.write(reinterpret_cast<const char*>(f_ext_0.data()), sizeof(double) * f_ext_0.size());
    //    buffer.write(reinterpret_cast<const char*>(m_ext_0.data()), sizeof(double) * m_ext_0.size());
    //    buffer.write(reinterpret_cast<const char*>(p.data()), sizeof(double) * p.size());
    //    buffer.write(reinterpret_cast<const char*>(p_0.data()), sizeof(double) * p_0.size());

    //    // Periodic flush to file (e.g., every 1000 iterations)
    //    if (buffer.str().size() > 4096) { // Write every 4KB of data
    //        File_data.write(buffer.str().c_str(), buffer.str().size());
    //        buffer.str("");  // Clear buffer
    //        buffer.clear();
    //    }


    // ************************************************************
    // YOUR CODE ENDS HERE
    // ************************************************************

    // For first iteration
    if( currentTime < sampleTime )
    {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }

    // A simple filter for the torque command
    tau_total = ( tau_motion + tau_previous + tau_prev_prev ) / 3;

    for ( int i=0; i<7; i++ )
    {
        qApplied[i] = filterOutput[i][0];
        torques[i] = tau_total[i];
    }

    // Command values (must be double arrays!)
    if (robotState().getClientCommandMode() == TORQUE)
    {
        robotCommand().setJointPosition(qApplied);
        robotCommand().setTorque(torques);
    }

    // IIR filter input
    iir(qCurr);

    // Updates
    if (currentTime == 0.0)
    {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }
    tau_previous = tau_motion;
    tau_prev_prev = tau_previous;

    currentTime = currentTime + sampleTime;

    p_avp_rw_prev = p_avp_rw;
    p_avp_rw_prev_prev = p_avp_rw_prev;

    R_avp_rw_prev = R_avp_rw;
    R_avp_rw_prev_prev = R_avp_rw_prev;

}


/**
* \brief Streamer Thread that polls the Vision Pro data and passes it to the command() loop
*
*/
void MyLBRClient::runStreamerThread() {
    try {
        // Create or open shared memory
        boost::interprocess::shared_memory_object shm(
            boost::interprocess::open_or_create, "SharedMemory_AVP", boost::interprocess::read_write);

        // Resize shared memory to hold a 4x4 double matrix (16 doubles, each 8 bytes) + version counter (8 bytes)
        // New: 6 * 16 doubles + ready flag 8 bytes = 6 * 16 * sizeof(double) + sizeof(int64_t)
        shm.truncate(1 * 16 * sizeof(double) + sizeof(int64_t));

        // Map the shared memory
        boost::interprocess::mapped_region region(shm, boost::interprocess::read_write);

        // Define pointers based on shared memory layout
        int64_t* ready_flag = reinterpret_cast<int64_t*>(region.get_address()); // First 8 bytes
        double* matrix_data_rw = reinterpret_cast<double*>(static_cast<char*>(region.get_address()) + sizeof(int64_t));                // First 4x4 matrix [0, :, :]

        // Wait for Python to initialize
        while (*ready_flag == -1) {
            std::cout << "Waiting for Python to initialize..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Python initialized. Starting processing..." << std::endl;

        int timeout_counter = 0;
        while (true) {
            if (*ready_flag == 1) {  // Check if Python has written new data

                dataMutex.lock();

                matrix_rw = matrix_data_rw;

                dataMutex.unlock();

                // Reset the flag
                *ready_flag = 0;
                timeout_counter = 0;  // Reset the timeout counter
            } else {
                timeout_counter++;
            }

            if (timeout_counter > 1000) {  // If nothing changes for a while
                timeout_counter = 0;  // Reset timeout to avoid permanent stop
            }

        }

    } catch (const std::exception& e) {
        std::cerr << "Error in shared memory operation: " << e.what() << std::endl;
    }
}



/**
* \brief Opens and runs the python script, stored locally
*
*/
void MyLBRClient::startPythonScript() {
    boost::thread pythonThread([]() {
        const std::string pythonScriptPath = "/home/newman_lab/Desktop/noah_repo/thesis-mit/AppleVisionPro/VisionProCppCommunication.py";
        //const std::string pythonScriptPath = "/../VisionProCppCommunication.py";
        const std::string pythonCommand = "python3 " + pythonScriptPath;

        int retCode = system(pythonCommand.c_str());
        if (retCode != 0) {
            std::cerr << "Error: Python script failed with return code " << retCode << std::endl;
        } else {
            std::cout << "Python script executed successfully." << std::endl;
        }
    });
    pythonThread.detach(); // Detach thread to run independently
}


/**
* \brief Thread that polls the force sensor data and passes it to the command() loop
*
*/
void MyLBRClient::forceSensorThread()
{
    while(true){

        double* ftsSignal = ftSensor->Acquire();

        //****************** Update everyting at the end with one Mutex ******************//
        mutexFTS.lock();

        f_sens_ee = ftsSignal;

        mutexFTS.unlock();


    }
}


/**
* \brief Function to compute damping factor, applied to stiffness matrix
*
*/
double MyLBRClient::compute_alpha(Eigen::Matrix3d& Lambda, Eigen::Vector3d& k_t, double damping_factor)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Lambda);
    Eigen::Matrix3d U = solver.eigenvectors();
    Eigen::Matrix3d Sigma = solver.eigenvalues().asDiagonal();

    for(int i=0; i<3; i++)
    {
        Sigma(i,i) = std::sqrt(Sigma(i,i));
    }

    // Compute sqrt(Lambda)
    //Eigen::Matrix3d sqrt_Lambda = U * Sigma.array().sqrt().matrix() * U.transpose();
    Eigen::Matrix3d sqrt_Lambda = U * Sigma * U.transpose();

    // Convert k_t to a diagonal matrix
    Eigen::Matrix3d sqrt_k_t = k_t.array().sqrt().matrix().asDiagonal();

    // Compute b_t
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity() * damping_factor;
    Eigen::Matrix3d b_t = sqrt_Lambda * D * sqrt_k_t + sqrt_k_t * D * sqrt_Lambda;

    // Compute alpha
    double alpha = (2.0 * b_t.trace()) / k_t.sum();
    return alpha;
}


/**
* \brief Function to compute damping factor, applied to stiffness matrix
*
*/
Eigen::MatrixXd MyLBRClient::getLambdaLeastSquares(Eigen::MatrixXd M, Eigen::MatrixXd J, double k)
{

    Eigen::MatrixXd Lambda_Inv = J * M.inverse() * J.transpose() + ( k * k ) * Eigen::MatrixXd::Identity( J.rows(), J.rows() );
    Eigen::MatrixXd Lambda = Lambda_Inv.inverse();
    return Lambda;

}











