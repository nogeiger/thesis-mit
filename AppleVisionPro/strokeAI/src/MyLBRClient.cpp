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
#include <ctime>
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
    // qInitial[0] = -8.87 * M_PI/180;
    // qInitial[1] = 60.98 * M_PI/180;
    // qInitial[2] = 17.51 * M_PI/180;
    // qInitial[3] = -79.85 * M_PI/180;
    // qInitial[4] = -24.13 * M_PI/180;
    // qInitial[5] = 43.03 * M_PI/180;
    // qInitial[6] = 4.14 * M_PI/180;

    qInitial[0] = -11.46 * M_PI/180;
    qInitial[1] = 95.12 * M_PI/180;
    qInitial[2] = 6.37 * M_PI/180;
    qInitial[3] = -66.35 * M_PI/180;
    qInitial[4] = 149.72 * M_PI/180;
    qInitial[5] = 70.83 * M_PI/180;
    qInitial[6] = 44.49 * M_PI/180;

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
    M = Eigen::MatrixXd::Zero( 7, 7 );
    M_inv = Eigen::MatrixXd::Zero( 7, 7 );

    pointPosition = Eigen::Vector3d( 0.0, 0.0, 0.0 );          // end-effector position
    //pointPosition = Eigen::Vector3d( 0.0, 0.0, 0.11 );      // with force sensor
    // pointPosition = Eigen::Vector3d( 0.0, 0.0, 0.16 );      // center of the hand palm

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

    J = Eigen::MatrixXd::Zero( 6, 7 );

    // Translational stiffness
    Kp = Eigen::MatrixXd::Identity( 3, 3 );                 // Translational stiffness
    Kr = Eigen::MatrixXd::Identity( 3, 3 );                 // Rotational stiffness

 
    // *********************************
    // Define stiffness categories here   
    std:string stiffness_cat = "medium";

    if (stiffness_cat == "low") {
        // Low stiffness
        printf("Selected stiffness category: low.\n\n");
        Kp = 350 * Kp;
        Kr = 30 * Kr;
    } 
    else if (stiffness_cat == "medium") {
        // Medium stiffness
        printf("Selected stiffness category: medium.\n\n");
        Kp = 650 * Kp;
        Kr = 100 * Kr;
    } 
    else if (stiffness_cat == "high") {
        // High stiffness
        printf("Selected stiffness category: high.\n\n");
        Kp = 850 * Kp;
        Kr = 150 * Kr;
    } 
    else {
        // Error message for invalid input
        printf("Error: Invalid stiffness category selected.\n\n");
    }
    
    // Joint space stiffness
    Kq = Eigen::MatrixXd::Identity( 7, 7 );                 // Nullpace stiffness (no need to change!)
    Kq = 10 * Kq;

    // Joint space damping
    Bq = Eigen::MatrixXd::Identity( 7, 7 );
    Bq = 0.5 * Bq;

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
    flag_hand = false;

    // ************************************************************
    // Store data
    // ************************************************************


    // Get the current timestamp and create the filename
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    
    char buffer[50];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", localTime); // Format: YYYY-MM-DD_HH-MM-SS
    
    std::string filename = "/home/newman_lab/Desktop/noah_repo/thesis-mit/AppleVisionPro/avp_stream/prints/DataHand_" 
                            + std::string(buffer) + ".bin";
    
    // Open a uniquely named binary file
    File_data.open(filename, std::ios::binary);
    if (!File_data) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
    } else {
        std::cout << "File successfully opened: " << filename << std::endl;
    }
    

    // ************************************************************
    // INCLUDE FT-SENSOR
    // ************************************************************
    // Weight: 0.2kg (plate) + 0.255kg (sensor) = 0.455kg

    f_ext_ee = Eigen::VectorXd::Zero( 3 );
    m_ext_ee = Eigen::VectorXd::Zero( 3 );
    f_ext = Eigen::VectorXd::Zero( 3 );
    m_ext = Eigen::VectorXd::Zero( 3 );

    AtiForceTorqueSensor ftSensor("172.31.1.1");
    
    // // Start threading for force sensor
    mutexFTS.unlock();
    boost::thread(&MyLBRClient::forceSensorThread, this).detach();

    printf( "Sensor Activated. \n\n" );


    // ************************************************************
    // ROBOTIC HAND INITIALIZATION
    // ************************************************************
    initializeRoboticHand();

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

    if (File_data.is_open()) {
        File_data.close();
    }

    // Shut down hand gracefully
    for (const auto& port : used_ports_) {
        comm_handler_->closeSerialPort(port);
    }

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
    //R_corrected.col(0) = R_avp_rw.col(0);           // X remains the same
    //R_corrected.col(1) = R_avp_rw.col(2);           // Z becomes Y (inverted)         
    //R_corrected.col(2) = -R_avp_rw.col(1);          // Y becomes Z

    R_corrected.col(0) = R_avp_rw.col(1);           // X gets Y
    R_corrected.col(1) = -R_avp_rw.col(2);           // Y becomes Z (inverted)         
    R_corrected.col(2) = -R_avp_rw.col(0);          // Z becomes X (inverted)

    R_avp_rw = R_corrected;

    // A simple filter for the rotation
    if( currentTime < sampleTime )
    {
        R_avp_rw_prev = R_avp_rw;
        R_avp_rw_prev_prev = R_avp_rw;
    }
    R_avp_rw = ( R_avp_rw + R_avp_rw_prev + R_avp_rw_prev_prev ) / 3;

    // Positon of knuckle with respect to avp
    //p_avp_rw = H_avp_rw.transpose().block< 3, 1 >( 0, 3 );
    p_avp_rw = H_avp_rw.transpose().block<3,1>(0,3)
             .cwiseProduct(Eigen::Vector3d(-1, -1, 1));




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
    f_ext = R * f_ext_ee;
    m_ext = R * m_ext_ee;

    // ************************************************************
    // Move Hand (flag_hand only)

    bool flag_override = false;

    // --- Apply flag_hand logic ---
    if (flag_hand && !hand_open) {
        hand_open = true;
        flag_override = true;
        std::cout << "[Hand] Opening due to flag_hand=true" << std::endl;
    } else if (!flag_hand && hand_open) {
        hand_open = false;
        flag_override = true;
        std::cout << "[Hand] Closing due to flag_hand=false" << std::endl;
    }

    // --- Apply new hand command only if needed ---
    if (flag_override) {
        for (const auto& id : device_ids_) {
            if (id.id == 0 || id.id == 120) continue;

            auto hand = soft_hands_.at(id.id);
            std::vector<int16_t> control_refs = hand_open ? std::vector<int16_t>{0} : std::vector<int16_t>{13000};

            hand->setMotorStates(true);
            hand->setControlReferences(control_refs);

            std::cout << "[Hand] Device " << (int)id.id << " set to " << (hand_open ? "OPEN" : "CLOSE") << std::endl;
        }
    }


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
    H = myLBR->getForwardKinematics( q, 7, pointPosition );
    R = H.block< 3, 3 >( 0, 0 );
    p = H.block< 3, 1 >( 0, 3 );

    // Transform rotations to quaternions
    Eigen::Matrix3d R_fixed = R.block<3,3>(0,0); // Ensure it's 3x3
    Eigen::Quaterniond Q(R_fixed);
    Q.normalize();

    // Extract rotation angle
    double theta = 2 * acos( Q.w() );
    double eps = 0.01;
    if( theta <  0.01 ){
        theta = theta + eps ;
    }
        
    // Compute norm factor, handle edge case for small theta
    double sin_half_theta  = sin(theta  / 2);
    double norm_fact ;
    if (fabs(sin_half_theta ) < 1e-6) {  // Handle small-angle case
        norm_fact  = 1.0;  // Default to 1, or handle separately
    } else {
        norm_fact  = 1.0 / sin_half_theta ;
    }
        
    Eigen::VectorXd u = Eigen::VectorXd::Zero( 3, 1 );
    u[0] = norm_fact  * Q.x();
    u[1] = norm_fact  * Q.y();
    u[2] = norm_fact  * Q.z();

    //  Get initial transfomation of first iteration
    if(currentTime < sampleTime)
    {
        H_ini = H;
        R_ini = R;
        p_ini = p;

        // Get initial AVP transformation
        p_avp_rw_ini = p_avp_rw;
        R_avp_rw_ini = R_avp_rw;

        // Get initial q
        q_ini = q;
    }

    // Jacobian, translational and rotation part
    J = myLBR->getHybridJacobian( q, pointPosition );
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

    // Apply scaling factor
    double scaleFactTranslation = 0.7; // Adjust as needed
    del_p_avp_rw /= scaleFactTranslation;

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
    Eigen::Quaterniond Q_ee_des(R_ee_des);
    Q_ee_des.normalize();

    // Extract rotation angle
    double theta_0 = 2 * acos( Q_ee_des.w() );
    double eps_0 = 0.01;
    if( theta_0 <  0.01 ){
        theta_0 = theta_0 + eps_0;
    }

    // Scale desired angle
    int scaleFact = 2;
    theta_0 = theta_0 / scaleFact;

    // Compute norm factor, handle edge case for small theta
    double sin_half_theta_0 = sin(theta_0 / 2);
    double norm_fact_0;
    if (fabs(sin_half_theta_0) < 1e-6) {  // Handle small-angle case
        norm_fact_0 = 1.0;  // Default to 1, or handle separately
    } else {
        norm_fact_0 = 1.0 / sin_half_theta_0;
    }

    Eigen::VectorXd u_ee = Eigen::VectorXd::Zero( 3, 1 );
    u_ee[0] = norm_fact_0 * Q_ee_des.x();
    u_ee[1] = norm_fact_0 * Q_ee_des.y();
    u_ee[2] = norm_fact_0 * Q_ee_des.z();

    // Transform to robot base coordinates
    Eigen::VectorXd u_0 = R * u_ee;

    // ************************************************************
    // Translational task-space impedance controller

    Eigen::VectorXd dx = J_v * dq;
    Eigen::VectorXd del_p = (p_0 - p);

    // Damping design
    double damping_factor_v = 0.7;
    Eigen::Vector3d Kp_diag = Kp.diagonal();
    Eigen::Matrix3d Lambda_v_3d = Lambda_v;
    double gamma_v = compute_gamma(Lambda_v_3d, Kp_diag, damping_factor_v);
    Eigen::MatrixXd Bp = gamma_v * Kp;

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
    Eigen::Matrix3d Lambda_w_3d = Lambda_w;
    double gamma_w = compute_gamma(Lambda_w_3d, Kr_diag, damping_factor_r);
    Eigen::MatrixXd Br = gamma_w * Kr;

    // Calculate moment
    Eigen::VectorXd m = Kr * u_0 * theta_0 - Br * omega;

    // Convert to torques
    Eigen::VectorXd tau_rotation = J_w.transpose() * m;


    // ************************************************************
    // Nullspace joint space stiffness
    Eigen::MatrixXd J_bar = M_inv * J.transpose() * Lambda;

    Eigen::MatrixXd N = Eigen::MatrixXd::Identity(7, 7) - J.transpose() * J_bar.transpose();

    Eigen::VectorXd tau_q = Bq * dq;

    // ************************************************************
    // Control torque
    tau_motion = tau_translation + tau_rotation + (N * tau_q);
    //tau_motion = tau_rotation + (N * tau_q);

    // Comment out for only gravity compensation
        //tau_motion = Eigen::VectorXd::Zero( myLBR->nq );

    // Include joint limits
    tau_motion = myLBR->addIIWALimits( q, dq, M, tau_motion, 0.004 );


    // ************************************************************
    // Write data in a file 

    // Buffer binary data
    buffer.write(reinterpret_cast<const char*>(&currentTime), sizeof(currentTime));
    buffer.write(reinterpret_cast<const char*>(f_ext.data()), sizeof(double) * f_ext.size());
    buffer.write(reinterpret_cast<const char*>(m_ext.data()), sizeof(double) * m_ext.size());
    buffer.write(reinterpret_cast<const char*>(p.data()), sizeof(double) * p.size());
    buffer.write(reinterpret_cast<const char*>(p_0.data()), sizeof(double) * p_0.size());
    buffer.write(reinterpret_cast<const char*>(u.data()), sizeof(double) * u.size());
    buffer.write(reinterpret_cast<const char*>(&theta), sizeof(double));  
    buffer.write(reinterpret_cast<const char*>(u_0.data()), sizeof(double) * u.size());
    buffer.write(reinterpret_cast<const char*>(&theta_0), sizeof(double)); 

    // Additionally for NLS
    buffer.write(reinterpret_cast<const char*>(dx.data()), sizeof(double) * dx.size());
    buffer.write(reinterpret_cast<const char*>(Lambda_v_3d.data()), sizeof(double) * Lambda_v_3d.size());
    buffer.write(reinterpret_cast<const char*>(omega.data()), sizeof(double) * omega.size());
    buffer.write(reinterpret_cast<const char*>(Lambda_w_3d.data()), sizeof(double) * Lambda_w_3d.size());

    // Periodic flush to file (e.g., every 1000 iterations)
    if (buffer.str().size() > 4096) { // Write every 4KB of data
        File_data.write(buffer.str().c_str(), buffer.str().size());
        buffer.str("");  // Clear buffer
        buffer.clear();
    }


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
            boost::interprocess::open_or_create, "SharedMemory_AVP_new", boost::interprocess::read_write);

        // Resize shared memory to hold a 4x4 double matrix (16 doubles, each 8 bytes) + version counter (8 bytes)
        // New: 6 * 16 doubles + ready flag 8 bytes = 6 * 16 * sizeof(double) + sizeof(int64_t)
        // 18 doubles (16 for 4x4 matrix + 2 flags) * 8 + 8 bytes for version flag = 152
        shm.truncate(1 * 17 * sizeof(double) + sizeof(int64_t));


        // Map the shared memory
        boost::interprocess::mapped_region region(shm, boost::interprocess::read_write);

        // Define pointers based on shared memory layout
        int64_t* ready_flag = reinterpret_cast<int64_t*>(region.get_address()); // First 8 bytes
        double* matrix_data_rw = reinterpret_cast<double*>(static_cast<char*>(region.get_address()) + sizeof(int64_t));                // First 4x4 matrix [0, :, :]

        int64_t* flag = reinterpret_cast<int64_t*>(static_cast<char*>(region.get_address()) + sizeof(int64_t) + sizeof(double) * 16);
        std::cout << "Flag value: " << *flag << std::endl;
          

        // Wait for Python to initialize
        while (*ready_flag == -1) {
            std::cout << "Waiting for Python to initialize...  \n\n" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Python initialized. Starting processing...  \n\n" << std::endl;

        int timeout_counter = 0;
        while (true) {
            if (*ready_flag == 1) {  // Check if Python has written new data

                dataMutex.lock();

                matrix_rw = matrix_data_rw;
                flag_hand = (*flag != 0);


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
double MyLBRClient::compute_gamma(Eigen::Matrix3d& Lam, Eigen::Vector3d& k_t, double damping_factor)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Lam);
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

    // Compute gamma
    double gamma = (2.0 * b_t.trace()) / k_t.sum();
    return gamma;
}


/**
* \brief Function to compute damping factor, applied to stiffness matrix
*
*/
Eigen::MatrixXd MyLBRClient::getLambdaLeastSquares(Eigen::MatrixXd M, Eigen::MatrixXd J, double k)
{
    Eigen::MatrixXd Lam_Inv = J * M.inverse() * J.transpose() + ( k * k ) * Eigen::MatrixXd::Identity( J.rows(), J.rows() );
    Eigen::MatrixXd Lam = Lam_Inv.inverse();

    return Lam;

}


/**
* \brief Function to initialize the robotic hand
*/
void MyLBRClient::initializeRoboticHand()
{
    hand_open = true;
    last_button_state = false;

    comm_handler_ = std::make_shared<qbrobotics_research_api::CommunicationLegacy>();
    std::vector<serial::PortInfo> serial_ports;

    if (comm_handler_->listSerialPorts(serial_ports) < 0) {
        std::cerr << "[qbHand] No serial ports found!" << std::endl;
        return;
    }

    for (const auto& port : serial_ports) {
        std::vector<qbrobotics_research_api::Communication::ConnectedDeviceInfo> ids;

        if (comm_handler_->openSerialPort(port.serial_port) >= 0) {
            used_ports_.insert(port.serial_port);

            if (comm_handler_->listConnectedDevices(port.serial_port, ids) >= 0) {
                std::cout << "[Init] Devices found on port: " << port.serial_port << std::endl;

                for (const auto& id : ids) {
                    if (id.id == 0 || id.id == 120) continue;

                    auto hand = std::make_shared<qbrobotics_research_api::qbSoftHandLegacyResearch>(
                        comm_handler_, "SoftHand", port.serial_port, id.id);

                    soft_hands_.insert({id.id, hand});
                    device_ids_.push_back(id);

                    std::vector<int16_t> open_ref = {0};
                    hand->setMotorStates(true);
                    hand->setControlReferences(open_ref);

                    std::cout << "  [Init] Device ID " << (int)id.id << " initialized and opened." << std::endl;
                }
            }
        }
    }
}
