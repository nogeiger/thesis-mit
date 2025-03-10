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
#include "AtiForceTorqueSensor.h"

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
    * If you do not want to change the default-behavior, you do not have to implement this method.
    */
    virtual void monitor();

    /**
    * \brief Callback for the FRI session state 'Commanding Wait'.
    *
    * If you do not want to change the default-behavior, you do not have to implement this method.
    */
    virtual void waitForCommand();


    /**
    * \brief Callback for the FRI state 'Commanding Active'.
    *
    * If you do not want to change the default-behavior, you do not have to implement this method.
    */
    virtual void command();


private:

    // Python Integration
    py::scoped_interpreter guard;

    // Shared Memory and Threading
    boost::thread streamerThread;
    boost::mutex dataMutex;

    void runStreamerThread();
    void startPythonScript();

    double* matrix_rw;

    Eigen::MatrixXd H_avp_rw_ini;
    Eigen::MatrixXd R_avp_rw_ini;
    Eigen::VectorXd p_avp_rw_ini;

    Eigen::MatrixXd p_avp_rw;
    Eigen::MatrixXd p_avp_rw_prev;
    Eigen::MatrixXd p_avp_rw_prev_prev;

    Eigen::MatrixXd R_avp_rw;
    Eigen::MatrixXd R_avp_rw_prev;
    Eigen::MatrixXd R_avp_rw_prev_prev;

    // Create iiwa as child of primitive class
    iiwa14 *myLBR;

    // Double values to get measured robot values and command robot values
    double torques[7];
    double qInitial[7];
    double xInitial[7];
    double qApplied[7];
    double qCurr[7];
    double qOld[7];
    double q_arr[7];
    double dq_arr[7];
    double tauExternal[7];

    // Time parameters for control loop
    double sampleTime;
    double currentTime;

    // Choose the body you want to control and the position on this body
    signed int bodyIndex;
    Eigen::Vector3d pointPosition;

    // Current position and velocity as Eigen vector
    Eigen::VectorXd q;
    Eigen::VectorXd q_ini;
    Eigen::VectorXd dq;

    // Command torque vectors (with and without constraints)
    Eigen::VectorXd tau_motion;
    Eigen::VectorXd tau_previous;
    Eigen::VectorXd tau_prev_prev;
    Eigen::VectorXd tau_total;

    // DECLARE VARIABLES FOR YOUR CONTROLLER HERE!!!
    Eigen::MatrixXd M;
    Eigen::MatrixXd M_inv;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
    Eigen::Matrix3d R_z;
    Eigen::Matrix3d R_avp_0;
    Eigen::MatrixXd J;
    
    Eigen::MatrixXd H_ini;
    Eigen::MatrixXd R_ini;
    Eigen::VectorXd p_ini;
    Eigen::VectorXd p;
    Eigen::VectorXd p_0_ini;
    
    Eigen::MatrixXd Kp;
    Eigen::MatrixXd Kr;
    Eigen::MatrixXd Kq;
    Eigen::MatrixXd Bq;

    // Damping will be calculated at runtime
    // Comment out for constant damping!
    //     Eigen::MatrixXd Bp;
    //     Eigen::MatrixXd Br;
    

    // Force-Torque Sensor
    AtiForceTorqueSensor *ftSensor;
    double* f_sens_ee;
    double* fts_first;
    Eigen::VectorXd f_ext_ee;
    Eigen::VectorXd m_ext_ee;
    Eigen::VectorXd f_ext;
    Eigen::VectorXd m_ext;
   

    void forceSensorThread();

    boost::thread ftsThread;
    boost::mutex mutexFTS;


    // Joint limit avoidance
    Eigen::VectorXd addConstraints(Eigen::VectorXd tauStack, double dt);
    double getMaxValue(Eigen::VectorXd myVector);
    double getMinValue(Eigen::VectorXd myVector);


    // Damping design
    double compute_alpha(Eigen::Matrix3d& Lambda, Eigen::Vector3d& k_t, double damping_factor);
    Eigen::MatrixXd getLambdaLeastSquares(Eigen::MatrixXd M, Eigen::MatrixXd J, double k);


    // Files to store data
    std::ostringstream buffer;
    std::ofstream File_data;


};

#endif // _KUKA_FRI_MY_LBR_CLIENT_H