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
        // Create or open a shared memory segment named "SharedMemory_AVP" with read/write permissions  
        boost::interprocess::shared_memory_object shm(  
            boost::interprocess::open_or_create, "SharedMemory_AVP", boost::interprocess::read_write);  

        // Resize the shared memory to accommodate:  
        // - A 4x4 matrix of doubles (16 doubles, each 8 bytes = 128 bytes)  
        // - A version counter (8 bytes, stored as int64_t)  
        shm.truncate(16 * sizeof(double) + sizeof(int64_t));  

        // Map the shared memory to the process's address space  
        boost::interprocess::mapped_region region(shm, boost::interprocess::read_write);  

        // Define pointers to access the shared memory layout  
        int64_t* ready_flag = reinterpret_cast<int64_t*>(region.get_address()); // First 8 bytes store the flag  
        double* matrix_data = reinterpret_cast<double*>(static_cast<char*>(region.get_address()) + sizeof(int64_t)); // Next 128 bytes store matrix  

        // Wait until Python sets the ready flag (i.e., no longer -1)  
        while (*ready_flag == -1) {  
            std::cout << "Waiting for Python to initialize..." << std::endl;  
            std::this_thread::sleep_for(std::chrono::milliseconds(10));  
        }  

        std::cout << "Python initialized. Starting processing..." << std::endl;  

        int timeout_counter = 0; // Counter to track the absence of updates  
        while (true) {  
            if (*ready_flag == 1) {  // Check if Python has written new data  
                dataMutex.lock();  // Lock to ensure thread safety  

                matrix = matrix_data; // Update the matrix with new shared memory data  

                dataMutex.unlock();  // Unlock after updating  

                // Reset the flag to indicate data has been processed  
                *ready_flag = 0;  
                timeout_counter = 0;  // Reset timeout counter after successful update  
            } else {  
                timeout_counter++;  // Increment timeout counter if no update occurs  
            }  

            if (timeout_counter > 1000) {  // If no updates are detected for a prolonged period   
                timeout_counter = 0;  // Reset timeout counter to prevent infinite waiting  
            }  
        }  

    } catch (const std::exception& e) {  
        // Catch and report any exceptions related to shared memory operations  
        std::cerr << "Error in shared memory operation: " << e.what() << std::endl;  
    }  
}  




//******************************************************************************  
// Function to Start the Python Script  
void MyLBRClient::startPythonScript() {  
    // Launch a new thread to run the Python script asynchronously  
    boost::thread pythonThread([]() {  
        // Define the path to the Python script  
        const std::string pythonScriptPath = "/home/newman_lab/Desktop/noah_repo/thesis-mit/VisionProCppCommunication.py";  

        // Construct the system command to execute the Python script  
        const std::string pythonCommand = "python3 " + pythonScriptPath;  

        // Execute the Python script using the system command  
        int retCode = system(pythonCommand.c_str());  

        // Check the return code to determine if the script executed successfully  
        if (retCode != 0) {  
            std::cerr << "Error: Python script failed with return code " << retCode << std::endl;  
        } else {  
            std::cout << "Python script executed successfully." << std::endl;  
        }  
    });  

    // Detach the thread so it runs independently without blocking execution  
    pythonThread.detach();  
}  



//******************************************************************************  
// Constructor for MyLBRClient class  
MyLBRClient::MyLBRClient(double freqHz, double amplitude)  
    : guard{} // Initialize guard for Python interpreter  
{  

    /** Initialization */  

    // Ensure this joint configuration matches the Java application  
    qInitial[0] = 12.58 * M_PI / 180;  
    qInitial[1] = 40.27 * M_PI / 180;  
    qInitial[2] = -0.01 * M_PI / 180;  
    qInitial[3] = -99.70 * M_PI / 180;  
    qInitial[4] = -0.01 * M_PI / 180;  
    qInitial[5] = 40.03 * M_PI / 180;  
    qInitial[6] = 12.59 * M_PI / 180;  

    // Create a new iiwa14 robot instance with ID 1 and name "Trey"  
    myLBR = new iiwa14(1, "Trey");  
    myLBR->init();  // Initialize the robot  

    // Initialize joint positions and velocities  
    q  = Eigen::VectorXd::Zero(myLBR->nq);  
    dq = Eigen::VectorXd::Zero(myLBR->nq);  

    // Time variables for the control loop  
    currentTime = 0;  
    sampleTime = 0;  
    t_pressed = 0; // Time elapsed after button press  

    // Initialize joint torques and joint positions (required for waitForCommand)  
    for (int i = 0; i < myLBR->nq; i++) {  
        qCurr[i] = qInitial[i]; // Set current joint positions to initial values  
        qOld[i] = qInitial[i];  // Store previous joint positions  
        qApplied[i] = 0.0;      // Initialize applied joint positions to zero  
        torques[i] = 0.0;       // Initialize torques to zero  
    }  

    // Initialize torque-related vectors  
    tau_motion    = Eigen::VectorXd::Zero(myLBR->nq);  
    tau_previous  = Eigen::VectorXd::Zero(myLBR->nq);  
    tau_prev_prev = Eigen::VectorXd::Zero(myLBR->nq);  
    tau_total     = Eigen::VectorXd::Zero(myLBR->nq);  

    // ************************************************************  
    // Initialize Vectors and Matrices  
    // ************************************************************  
    M = Eigen::MatrixXd::Zero(myLBR->nq, myLBR->nq);      // Mass matrix  
    M_inv = Eigen::MatrixXd::Zero(myLBR->nq, myLBR->nq);  // Inverse of mass matrix  

    // Initialize 3D point position  
    pointPosition[0] = 0.0;  
    pointPosition[1] = 0.0;  
    pointPosition[2] = 0.085;  

    bodyIndex = 7; // Index for tracking the robot's body  

    // Homogeneous transformation matrices  
    H = Eigen::MatrixXd::Zero(4, 4);  
    R = Eigen::MatrixXd::Zero(3, 3);  
    R_ee_i = Eigen::Matrix3d::Zero(3, 3);  

    // Rotation matrix around Z-axis  
    R_z <<  0.0, -1.0,  0.0,  
            1.0,  0.0,  0.0,  
            0.0,  0.0,  1.0;  

    // Initialize transformation matrices and position vectors  
    H_ini = Eigen::MatrixXd::Zero(4, 4);  
    R_ini = Eigen::MatrixXd::Zero(3, 3);  
    p_ini = Eigen::VectorXd::Zero(3, 1);  
    p_0_ini = Eigen::VectorXd::Zero(3, 1);  
    p_vp_3d = Eigen::VectorXd::Zero(3, 1);  
    p = Eigen::VectorXd::Zero(3, 1);  

    // Jacobian matrix initialization  
    J = Eigen::MatrixXd::Zero(6, myLBR->nq);  

    // Translational impedance control parameters  
    Kp = Eigen::MatrixXd::Identity(3, 3);  
    Kp = 900 * Kp; // Translational stiffness  
    Bp = Eigen::MatrixXd::Identity(3, 3);  
    Bp = 70 * Bp;  // Translational damping  

    // Rotational impedance control parameters  
    Kr = Eigen::MatrixXd::Identity(3, 3);  
    Kr = 150 * Kr; // Rotational stiffness  
    Br = Eigen::MatrixXd::Identity(3, 3);  
    Br = 8 * Br;   // Rotational damping  

    // ************************************************************  
    // AVP Streamer Initialization  
    // ************************************************************  

    // Unlock the mutex to allow data streaming  
    dataMutex.unlock();  

    // Launch the streamer thread asynchronously  
    boost::thread(&MyLBRClient::runStreamerThread, this).detach();  

    // Start the Python script responsible for additional processing  
    startPythonScript();  

    // Initialize AVP transformation matrices  
    H_rw_ini = Eigen::MatrixXd::Identity(4, 4);  
    p_rw_ini = Eigen::VectorXd::Zero(3);  

    // Allocate memory for a 4x4 matrix (16 elements)  
    matrix = new double[16];  

    // ************************************************************  
    // Initial Print Statement  
    // ************************************************************  
    printf("Exp[licit](c)-cpp-FRI, https://explicit-robotics.github.io \n\n");  
    printf("Robot '");  
    printf("%s", myLBR->Name);  
    printf("' initialized. Ready to rumble! \n\n");  
}  


/**
 * \brief Destructor for MyLBRClient
 *
 * Cleans up allocated resources and removes shared memory.
 */
MyLBRClient::~MyLBRClient()
{
    // Remove the shared memory segment named "SharedMemory_AVP"
    boost::interprocess::shared_memory_object::remove("SharedMemory_AVP");

    // Delete dynamically allocated robot instance to prevent memory leaks
    delete myLBR;
}

/**
 * \brief Implements an IIR (Infinite Impulse Response) Filter
 *
 * This filter is used to send the previous joint position to the command function, 
 * ensuring that KUKA's internal friction compensation is activated. 
 * The filter function was generated using the application WinFilter (http://www.winfilter.20m.com/).
 *
 * @param NewSample The current joint position array (size 7) provided as input to the filter.
 */
void iir(double NewSample[7])
{
    // Coefficients for the IIR filter (generated using WinFilter)
    double ACoef[NCoef+1] = {
        0.05921059165970496400,  // Coefficient A0
        0.05921059165970496400   // Coefficient A1
    };

    double BCoef[NCoef+1] = {
        1.00000000000000000000,   // Coefficient B0 (always 1 for normalization)
        -0.88161859236318907000   // Coefficient B1 (feedback coefficient)
    };

    int n;

    // Shift old filter samples to make room for the new sample
    for (int i = 0; i < 7; i++)
    {
        for (n = NCoef; n > 0; n--)
        {
            filterInput[i][n] = filterInput[i][n-1];   // Shift input history
            filterOutput[i][n] = filterOutput[i][n-1]; // Shift output history
        }
    }

    // Compute the new filtered output for each joint
    for (int i = 0; i < 7; i++)
    {
        filterInput[i][0] = NewSample[i];  // Store the new input sample
        filterOutput[i][0] = ACoef[0] * filterInput[i][0]; // Initialize output with first coefficient
    }

    // Apply the IIR filter equation
    for (int i = 0; i < 7; i++)
    {
        for (n = 1; n <= NCoef; n++)
        {
            filterOutput[i][0] += ACoef[n] * filterInput[i][n] - BCoef[n] * filterOutput[i][n];
        }
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
// Function to monitor the robot's state and update joint positions  
void MyLBRClient::monitor()  
{  
    // Copy the commanded joint position to the robot command  
    robotCommand().setJointPosition(robotState().getCommandedJointPosition());  

    // Copy the measured joint positions (radians) into the qCurr array  
    memcpy(qCurr, robotState().getMeasuredJointPosition(), 7 * sizeof(double));  

    // Initialize the joint positions for the previous NCoef timesteps  
    for (int i = 0; i < NCoef + 1; i++)  
    {  
        iir(qCurr);  // Apply IIR filtering to smooth joint positions  
    }  
}  

//******************************************************************************  
// Function to wait for a command from the robot  
void MyLBRClient::waitForCommand()  
{  
    // If in torque mode, a torque command must be sent continuously  
    if (robotState().getClientCommandMode() == TORQUE)  
    {  
        robotCommand().setTorque(torques);  // Send torque values  
        robotCommand().setJointPosition(robotState().getIpoJointPosition()); // Maintain current position  
    }  
}  

//******************************************************************************  
// Function to compute and send commands to the robot  
void MyLBRClient::command()  
{  
    // ************************************************************  
    // Read AVP-relative positions  

    if (currentTime < sampleTime)  
    {  
        startPythonScript(); // Restart Python script if needed  
    }  

    double* test_matrix;  
    Eigen::MatrixXd H_rw;  
    Eigen::VectorXd p_rw;  

    // Lock mutex to safely access shared memory  
    dataMutex.lock();  
    test_matrix = matrix;  
    dataMutex.unlock();  

    // Map shared memory data into an Eigen 4x4 matrix  
    H_rw = Eigen::Map<Eigen::MatrixXd>(test_matrix, 4, 4);  
    p_rw = H_rw.transpose().block<3, 1>(0, 3);  

    // ************************************************************  
    // Get robot measurements  

    // Store previous joint positions  
    memcpy(qOld, qCurr, 7 * sizeof(double));  
    // Update current joint positions and external torques  
    memcpy(qCurr, robotState().getMeasuredJointPosition(), 7 * sizeof(double));  
    memcpy(tauExternal, robotState().getExternalTorque(), 7 * sizeof(double));  

    // Copy joint positions to Eigen vector  
    for (int i = 0; i < myLBR->nq; i++)  
    {  
        q[i] = qCurr[i];  
    }  

    // Compute joint velocities  
    for (int i = 0; i < 7; i++)  
    {  
        dq[i] = (qCurr[i] - qOld[i]) / sampleTime;  
    }  

    // Capture initial transformation if first iteration  
    if (currentTime < sampleTime)  
    {  
        H_ini = myLBR->getForwardKinematics(q);  
        R_ini = H_ini.block<3, 3>(0, 0);  
        p_ini = H_ini.block<3, 1>(0, 3);  

        // Save initial AVP position  
        H_rw_ini = H_rw;  
        p_rw_ini = H_rw_ini.transpose().block<3, 1>(0, 3);  
    }  

    // ************************************************************  
    // Compute kinematics and dynamics  

    // Compute forward kinematics (transformation matrix)  
    H = myLBR->getForwardKinematics(q);  
    R = H.block<3, 3>(0, 0);  
    p = H.block<3, 1>(0, 3);  

    // Compute Jacobian matrix  
    J = myLBR->getHybridJacobian(q);  
    Eigen::MatrixXd J_v = J.block(0, 0, 3, 7); // Translational Jacobian  
    Eigen::MatrixXd J_w = J.block(3, 0, 3, 7); // Rotational Jacobian  

    // Adjust mass matrix to reduce acceleration at the last joint  
    M = myLBR->getMassMatrix(q);  
    M(6, 6) = 40 * M(6, 6);  
    M_inv = M.inverse();  

    // ************************************************************  
    // Compute position differences  

    // Compute displacement from initial AVP position  
    Eigen::VectorXd p_vp_3d = p_rw - p_rw_ini;  

    // Convert to homogeneous coordinates  
    Eigen::VectorXd p_vp_4d = Eigen::VectorXd::Zero(4, 1);  
    p_vp_4d[3] = 1;  
    p_vp_4d.head<3>() = p_vp_3d;  

    // Transform AVP coordinates to robot base frame  
    Eigen::MatrixXd H_rel = Eigen::MatrixXd::Zero(4, 4);  
    H_rel.block<3, 3>(0, 0) = R_z;  
    H_rel.block<3, 1>(0, 3) = p_ini;  

    Eigen::VectorXd p_0_4d = H_rel * p_vp_4d;  
    Eigen::VectorXd p_0 = p_0_4d.block<3, 1>(0, 0);  

    // Compute rotation error  
    Eigen::Matrix3d R_ee_des = R.transpose() * R_ini;  
    Eigen::Quaterniond Q(R_ee_des);  
    Q.normalize();  
    double theta = 2 * acos(Q.w());  

    // Prevent division by zero  
    double eps = 0.01;  
    if (theta < 0.01)  
    {  
        theta += eps;  
    }  

    double norm_fact = 1 / sin(theta / 2);  
    Eigen::VectorXd u_ee = Eigen::VectorXd::Zero(3, 1);  
    u_ee[0] = norm_fact * Q.x();  
    u_ee[1] = norm_fact * Q.y();  
    u_ee[2] = norm_fact * Q.z();  

    // ************************************************************  
    // Compute control torques  

    // Translational Cartesian impedance controller  
    Eigen::VectorXd dx = J_v * dq;  
    Eigen::VectorXd del_p = (p_0 - p);  
    Eigen::VectorXd f = Kp * del_p - Bp * dx;  
    Eigen::VectorXd tau_translation = J_v.transpose() * f;  

    // Rotational impedance controller  
    Eigen::VectorXd omega = J_w * dq;  
    Eigen::VectorXd u_0 = R * u_ee;  
    Eigen::VectorXd tau_rotation = J_w.transpose() * (Kr * u_0 * theta - Br * omega);  

    // Compute total control torque  
    tau_motion = tau_translation + tau_rotation;  

    // Apply joint limits  
    tau_motion = myLBR->addIIWALimits(q, dq, M, tau_motion, 0.004);  

    // Apply a simple filter to smooth the torque command  
    tau_total = (tau_motion + tau_previous + tau_prev_prev) / 3;  

    for (int i = 0; i < 7; i++)  
    {  
        qApplied[i] = filterOutput[i][0];  
        torques[i] = tau_total[i];  
    }  

    // Send torque and joint position commands  
    if (robotState().getClientCommandMode() == TORQUE)  
    {  
        robotCommand().setJointPosition(qApplied);  
        robotCommand().setTorque(torques);  
    }  

    // Update IIR filter with current joint positions  
    iir(qCurr);  

    // Update previous torques  
    if (currentTime == 0.0)  
    {  
        tau_previous = tau_motion;  
        tau_prev_prev = tau_motion;  
    }  
    tau_previous = tau_motion;  
    tau_prev_prev = tau_previous;  

    // Increment time step  
    currentTime += sampleTime;  
}  




