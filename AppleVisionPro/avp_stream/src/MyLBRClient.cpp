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


//******************************************************************************
void MyLBRClient::runStreamerThread() {
    try {
        // Create or open shared memory
        boost::interprocess::shared_memory_object shm(
            boost::interprocess::open_or_create, "SharedMemory_AVP", boost::interprocess::read_write);

        // Resize shared memory to hold a 4x4 double matrix (16 doubles, each 8 bytes) + version counter (8 bytes)
        shm.truncate(16 * sizeof(double) + sizeof(int64_t));

        // Map the shared memory
        boost::interprocess::mapped_region region(shm, boost::interprocess::read_write);

        // Define pointers based on shared memory layout
        int64_t* ready_flag = reinterpret_cast<int64_t*>(region.get_address()); // First 8 bytes
        double* matrix_data = reinterpret_cast<double*>(static_cast<char*>(region.get_address()) + sizeof(int64_t)); // Next 128 bytes

        // Wait for Python to initialize
        while (*ready_flag == -1) {
            std::cout << "Waiting for Python to initialize..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Python initialized. Starting processing..." << std::endl;

        int timeout_counter = 0;
        while (true) {
            if (*ready_flag == 1) {  // Check if Python has written new data
                //                std::cout << "C++ detected Ready flag: " << *ready_flag << std::endl;

                // Read and process the matrix
                //                std::cout << "C++ read Matrix data:" << std::endl;
                //                for (int i = 0; i < 4; i++) {
                //                    for (int j = 0; j < 4; j++) {
                //                        std::cout << matrix_data[i * 4 + j] << " ";
                //                    }
                //                    std::cout << std::endl;
                //                }

                // Be carful, Johannes was there
                dataMutex.lock();

                matrix = matrix_data;

                dataMutex.unlock();

                // Reset the flag
                *ready_flag = 0;
                //                std::cout << "C++ reset Ready flag to: " << *ready_flag << std::endl;
                timeout_counter = 0;  // Reset the timeout counter
            } else {
                timeout_counter++;
            }

            if (timeout_counter > 1000) {  // If nothing changes for a while
                //                std::cout << "No updates from Python detected, continuing..." << std::endl;
                timeout_counter = 0;  // Reset timeout to avoid permanent stop
            }

            // Sleep briefly
            //std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in shared memory operation: " << e.what() << std::endl;
    }
}



//******************************************************************************
// Start Python Script
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


//******************************************************************************
//MyLBRClient::MyLBRClient(double freqHz, double amplitude){

MyLBRClient::MyLBRClient(double freqHz, double amplitude)
    :guard{} // Initialize guard (Python interpreter)
{

    /** Initialization */

    // THIS CONFIGURATION MUST BE THE SAME AS FOR THE JAVA APPLICATION!!
    qInitial[0] = 12.58 * M_PI/180;
    qInitial[1] = 40.27 * M_PI/180;
    qInitial[2] = -0.01 * M_PI/180;
    qInitial[3] = -99.70 * M_PI/180;
    qInitial[4] = -0.01 * M_PI/180;
    qInitial[5] = 40.03 * M_PI/180;
    qInitial[6] = 12.59 * M_PI/180;

    // Use Explicit-cpp to create your robot
    myLBR = new iiwa14( 1, "Trey");
    myLBR->init( );

    // Current joint configuration and velocity
    q  = Eigen::VectorXd::Zero( myLBR->nq );
    dq = Eigen::VectorXd::Zero( myLBR->nq );

    // Time variables for control loop
    currentTime = 0;
    sampleTime = 0;
    t_pressed = 0;      // Time after the button push

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
    R_ee_i = Eigen::Matrix3d::Zero( 3, 3 );

    R_z <<  0.0, -1.0,  0.0,
        1.0, 0.0,  0.0,
        0.0, 0.0, 1.0;

    H_ini = Eigen::MatrixXd::Zero( 4, 4 );
    R_ini = Eigen::MatrixXd::Zero( 3, 3 );
    p_ini = Eigen::VectorXd::Zero( 3, 1 );
    p_0_ini = Eigen::VectorXd::Zero( 3, 1 );
    p_vp_3d = Eigen::VectorXd::Zero( 3, 1 );
    p = Eigen::VectorXd::Zero( 3, 1 );

    J = Eigen::MatrixXd::Zero( 6, myLBR->nq );

    // Translational impedances
    Kp = Eigen::MatrixXd::Identity( 3, 3 );
    Kp = 900 * Kp;
    Bp = Eigen::MatrixXd::Identity( 3, 3 );
    Bp = 70 * Bp;

    // Rotational impedances
    Kr = Eigen::MatrixXd::Identity( 3, 3 );
    Kr = 150 * Kr;
    Br = Eigen::MatrixXd::Identity( 3, 3 );
    Br = 8 * Br;

    // ************************************************************
    // AVP streamer
    // ************************************************************

    // Lock mutex initially
    dataMutex.unlock();
    boost::thread(&MyLBRClient::runStreamerThread, this).detach();

    // Start the Python script
    startPythonScript();

    // Transformation matrices of AVP
    H_rw_ini = Eigen::MatrixXd::Identity( 4, 4 );
    p_rw_ini = Eigen::VectorXd::Zero( 3 );

    matrix = new double[16];

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

    double* test_matrix;
    Eigen::MatrixXd H_rw;
    Eigen::VectorXd p_rw;

    // Lock mutex and update local variables from shared memory
    dataMutex.lock();

    test_matrix = matrix;

    dataMutex.unlock();

    H_rw = Eigen::Map<Eigen::MatrixXd>(test_matrix, 4, 4);
    p_rw = H_rw.transpose().block< 3, 1 >( 0, 3 );

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

    //  Get initial transfomation of first iteration
    if(currentTime < sampleTime)
    {
        H_ini = myLBR->getForwardKinematics( q );
        R_ini = H_ini.block< 3, 3 >( 0, 0 );
        p_ini = H_ini.block< 3, 1 >( 0, 3 );

        // ****************** GET INITIAL STREAM POSITION ******************
        H_rw_ini = H_rw;
        p_rw_ini = H_rw_ini.transpose().block< 3, 1 >( 0, 3 );
    }

    // ************************************************************
    // Calculate kinematics and dynamics

    // Transformation and Rotation Matrix
    //    H = myLBR->getForwardKinematics( q, bodyIndex, pointPosition );
    H = myLBR->getForwardKinematics( q );
    R = H.block< 3, 3 >( 0, 0 );
    p = H.block< 3, 1 >( 0, 3 );

    // Jacobian, translational and rotation part
    //J = myLBR->getHybridJacobian( q, pointPosition );
    J = myLBR->getHybridJacobian( q );
    Eigen::MatrixXd J_v = J.block(0, 0, 3, 7);
    Eigen::MatrixXd J_w = J.block(3, 0, 3, 7);

    // Adapt mass matrix to prevent high accelerations at last joint
    M = myLBR->getMassMatrix( q );
    M( 6, 6 ) = 40 * M( 6, 6 );
    M_inv = this->M.inverse();

    // ****************** ADAPT POSITION DIFFERENCE ******************

    // Proceed with your control logic using H_rw and p_rw
    Eigen::VectorXd p_vp_3d = p_rw - p_rw_ini;

    // Transform to homogeneous coordinates
    Eigen::VectorXd p_vp_4d = Eigen::VectorXd::Zero(4, 1);
    p_vp_4d[3] = 1;
    p_vp_4d.head<3>() = p_vp_3d;

    // Transformation to spatail robot coordinates
    Eigen::MatrixXd H_rel = Eigen::MatrixXd::Zero( 4, 4 );
    H_rel.block<3, 3>(0, 0) = R_z;
    H_rel.block<3, 1>(0, 3) = p_ini;

    // Extract 3x1 position
    Eigen::VectorXd p_0_4d = H_rel * p_vp_4d;
    Eigen::VectorXd p_0 = p_0_4d.block<3, 1>(0, 0);

    // Rotations
    Eigen::Matrix3d R_ee_des = R.transpose() * R_ini;
    Eigen::Quaterniond Q(R_ee_des);         // Transform to quaternions
    Q.normalize();
    double theta = 2 * acos( Q.w() );

    double eps = 0.01;
    if( theta <  0.01 ){
        theta = theta + eps;
    }

    double norm_fact = 1 / sin( theta/2 );

    Eigen::VectorXd u_ee = Eigen::VectorXd::Zero( 3, 1 );
    u_ee[0] = norm_fact * Q.x();
    u_ee[1] = norm_fact * Q.y();
    u_ee[2] = norm_fact * Q.z();


    // ************************************************************
    // Control torque

    // Translational Cartesian impedance controller
    Eigen::VectorXd dx = J_v * dq;

    Eigen::VectorXd del_p = (p_0 - p);

    Eigen::VectorXd f = Kp * del_p - Bp * dx;

    Eigen::VectorXd tau_translation = J_v.transpose() * f;

    // Rotational impedance controler
    Eigen::VectorXd omega = J_w * dq;

    Eigen::VectorXd u_0 = R * u_ee;

    Eigen::VectorXd tau_rotation = J_w.transpose() * ( Kr * u_0 * theta - Br * omega );

    // Control torque
    tau_motion = tau_translation + tau_rotation;

    // Just gravity compensation
    //    tau_motion = Eigen::VectorXd::Zero( myLBR->nq );

    // Include joint limits
    tau_motion = myLBR->addIIWALimits( q, dq, M, tau_motion, 0.004 );


    // ************************************************************
    // YOUR CODE ENDS HERE!
    // ************************************************************

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

    // Update
    if (currentTime == 0.0)
    {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }
    tau_previous = tau_motion;
    tau_prev_prev = tau_previous;


    // Print stuff (later if needed)
    if( currentTime < sampleTime )
    {
        //        DO STUFF
    }

    currentTime = currentTime + sampleTime;

}





