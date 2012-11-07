#include <main.cu>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

void uniform_01(vector<float> &f) {
    for(int i=0; i<f.size(); i++) {
        f[i]=(float) rand()/(float) RAND_MAX;
    }
}

float uniform_rand01() {
    


    return 500*(float) rand()/(float) RAND_MAX;
}

void load_in_rands(vector<float> &r1, 
                   vector<float> &r2, 
                   vector<float> &r3, 
                   char *filename) {

    ifstream in(filename);   

    string line;
    int count = 0;
    while(getline(in, line)) {
        stringstream ss(line);
        ss >> r1[count];
        ss >> r2[count];
        ss >> r3[count];
        count++;
    }

    assert(count == r1.size());

    in.close();

}


void load_in_r(vector<float> &rx,
               vector<float> &ry,
               vector<float> &rz,
               vector<int> &id,
               char *filename) {

    ifstream in(filename);   

    string line;
    int count = 0;
    while(getline(in, line)) {
        stringstream ss(line);
        ss >> rx[count];
        ss >> ry[count];
        ss >> rz[count];
        ss >> id[count];
        count++;
    }

    assert(count == rx.size());

    in.close();
}

void load_in_q(vector<float> &qx, 
               vector<float> &qy,
               vector<float> &qz,
               char *filename) {
    ifstream in(filename);   

    string line;
    int count = 0;
    while(getline(in, line)) {
        stringstream ss(line);
        ss >> qx[count];
        ss >> qy[count];
        ss >> qz[count];
        //ss >> oq[count];
        count++;
    }


    assert(count == qx.size());

    in.close();
}


void deviceMalloc( void ** ptr, int bytes) {
    cudaError_t err = cudaMalloc(ptr, (size_t) bytes);
    assert(err == 0);
} 


inline ostream& operator<< (ostream &out, const vector<int> &s) {
    for(int i=0; i<s.size(); i++) {
        out << s[i] << " ";
    }
    out << endl;
    return out;
}


inline ostream& operator<< (ostream &out, const vector<float> &s) {
    for(int i=0; i<s.size(); i++) {
        out << s[i] << " ";
    }
    out << endl;   
    return out;
}

int main() {

    // allocate qx, qy, qz, outQ, |nQ|
    // allocate rx, ry, rz, |numAtoms|
    // allocate atomicIdentities

    const int tpb = 512;
    const int bpg = 4;

    int nRotations = tpb*bpg;
    int nAtoms = 512; 
    int nQ = 512;

    unsigned int nQ_size = nQ*sizeof(int);
    unsigned int nAtoms_size = nAtoms*sizeof(int);
    unsigned int nRotations_size = nRotations*sizeof(int);

    vector<float> h_qx(nQ);
    vector<float> h_qy(nQ);
    vector<float> h_qz(nQ);
    vector<float> h_outQ(nQ);
    vector<float> h_rx(nAtoms);
    vector<float> h_ry(nAtoms);
    vector<float> h_rz(nAtoms);
    vector<int>   h_id(nAtoms);

    vector<float> h_rand1(nRotations);
    vector<float> h_rand2(nRotations);
    vector<float> h_rand3(nRotations);

    load_in_r(h_rx, h_ry, h_rz, h_id, "512_atom_benchmark.xyz");
    load_in_q(h_qx, h_qy, h_qz, "512_q.xyz");
    load_in_rands(h_rand1, h_rand2, h_rand3, "2048_x_3_random_floats.txt");

    for(int i=0; i < h_outQ.size(); i++) {
        h_outQ[i]=0;
    } 

    float *d_qx; deviceMalloc( (void **) &d_qx, nQ_size);
    float *d_qy; deviceMalloc( (void **) &d_qy, nQ_size);
    float *d_qz; deviceMalloc( (void **) &d_qz, nQ_size);
    float *d_outQ; deviceMalloc( (void **) &d_outQ, nQ_size);
    float *d_rx; deviceMalloc( (void **) &d_rx, nAtoms_size);
    float *d_ry; deviceMalloc( (void **) &d_ry, nAtoms_size);
    float *d_rz; deviceMalloc( (void **) &d_rz, nAtoms_size);
    int   *d_id; deviceMalloc( (void **) &d_id, nAtoms_size);
    float *d_rand1; deviceMalloc( (void **) &d_rand1, nRotations_size);
    float *d_rand2; deviceMalloc( (void **) &d_rand2, nRotations_size);
    float *d_rand3; deviceMalloc( (void **) &d_rand3, nRotations_size);

    cudaMemcpy(d_qx, &h_qx[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &h_qy[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &h_qz[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outQ, &h_outQ[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx, &h_rx[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, &h_ry[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, &h_rz[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_id, &h_id[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand1, &h_rand1[0], nRotations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand2, &h_rand2[0], nRotations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand3, &h_rand3[0], nRotations_size, cudaMemcpyHostToDevice);

    cudaError_t error;

    kernel<tpb> <<<bpg, tpb>>> (d_qx, d_qy, d_qz, d_outQ, nQ, d_rx, d_ry, d_rz, d_id, nAtoms, d_rand1, d_rand2, d_rand3);

    cudaThreadSynchronize();
    error = cudaGetLastError(); printf("Last error: %d \n", error);

    cudaMemcpy(&h_outQ[0], d_outQ, nQ_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    error = cudaGetLastError(); printf("Last error: %d \n", error);

    for(int i=0; i<min((int)h_outQ.size(),512); i++) {
        printf("%.0f ", h_outQ[i]);
    }

}
