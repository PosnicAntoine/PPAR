/*
 * Conway's Game of Life
 * A. Mucherino
 * PPAR, TP5
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

int N = 32;
int itMax = 200;

// allocation only
unsigned int* allocate(){
    return (unsigned int*)calloc(N*N,sizeof(unsigned int));
}

// conversion cell location : 2d --> 1d
// (row by row)
int code(int x,int y,int dx,int dy){
    int i = (x + dx)%N;
    int j = (y + dy)%N;
    if (i < 0)  i = N + i;
    if (j < 0)  j = N + j;
    return i*N + j;
}

// writing into a cell location
void write_cell(int x,int y,unsigned int value,unsigned int *world){
    int k = code(x,y,0,0);
    world[k] = value;
}

// random generation
unsigned int* initialize_random(){
    int x,y;
    unsigned int cell;
    unsigned int *world;
    world = allocate();
    for (x = 0; x < N; x++){
        for (y = 0; y < N; y++){
            if (rand()%5){
                cell = 0;
            }else{
                cell = rand()%2?2:1;
            }
            write_cell(x,y,cell,world);
        }
    }
    return world;
}

// dummy generation
unsigned int* initialize_dummy(){
    int x,y;
    unsigned int *world;

    world = allocate();
    for (x = 0; x < N; x++){
        for (y = 0; y < N; y++){
            write_cell(x,y,x%3,world);
        }
    }
    return world;
}

// "glider" generation
unsigned int* initialize_glider(){
    int x,y,mx,my;
    unsigned int *world;

    world = allocate();
    for (x = 0; x < N; x++){
        for (y = 0; y < N; y++){
            write_cell(x,y,0,world);
        }
    }
    mx = N/2 - 1;  my = N/2 - 1;
    x = mx;      y = my + 1;  write_cell(x,y,1,world);
    x = mx + 1;  y = my + 2;  write_cell(x,y,1,world);
    x = mx + 2;  y = my;      write_cell(x,y,1,world);
                 y = my + 1;  write_cell(x,y,1,world);
                 y = my + 2;  write_cell(x,y,1,world);

    return world;
}

// "small exploder" generation
unsigned int* initialize_small_exploder(){
    int x,y,mx,my;
    unsigned int *world;

    world = allocate();
    for (x = 0; x < N; x++){
        for (y = 0; y < N; y++){
            write_cell(x,y,0,world);
        }
    }

    mx = N/2 - 2;    my = N/2 - 2;
    x = mx;      y = my + 1;  write_cell(x,y,2,world);
    x = mx + 1;  y = my;      write_cell(x,y,2,world);
                 y = my + 1;  write_cell(x,y,2,world);
                 y = my + 2;  write_cell(x,y,2,world);
    x = mx + 2;  y = my;      write_cell(x,y,2,world);
                 y = my + 2;  write_cell(x,y,2,world);
    x = mx + 3;  y = my + 1;  write_cell(x,y,2,world);

    return world;
}


// reading a cell
int read_cell(int x,int y,int dx,int dy,unsigned int *world){
    int k = code(x,y,dx,dy);
    return world[k];
}

// updating counters
void update(int x,int y,int dx,int dy,unsigned int *world,int *nn,int *n1,int *n2){
    unsigned int cell = read_cell(x,y,dx,dy,world);
    if (cell){
        (*nn)++;
        if (cell == 1){
            (*n1)++;
        }else{
            (*n2)++;
        }
    }
}

// looking around the cell
void neighbors(int x,int y,unsigned int *world,int *nn,int *n1,int *n2){
    int dx,dy;

    (*nn) = 0;  (*n1) = 0;  (*n2) = 0;

    // same line
    dx = -1;  dy = 0;   update(x,y,dx,dy,world,nn,n1,n2);
    dx = +1;  dy = 0;   update(x,y,dx,dy,world,nn,n1,n2);

    // one line down
    dx = -1;  dy = +1;  update(x,y,dx,dy,world,nn,n1,n2);
    dx =  0;  dy = +1;  update(x,y,dx,dy,world,nn,n1,n2);
    dx = +1;  dy = +1;  update(x,y,dx,dy,world,nn,n1,n2);

    // one line up
    dx = -1;  dy = -1;  update(x,y,dx,dy,world,nn,n1,n2);
    dx =  0;  dy = -1;  update(x,y,dx,dy,world,nn,n1,n2);
    dx = +1;  dy = -1;  update(x,y,dx,dy,world,nn,n1,n2);
}

// computing a new generation
short newgeneration(unsigned int *world1,unsigned int *world2,int xstart,int xend){
    int x,y;
    int nn,n1,n2;
    unsigned int cell;
    unsigned int nextvalue;
    short change = 0;

    // cleaning destination world
    for (x = 0; x < N; x++){
        for (y = 0; y < N; y++){
            write_cell(x,y,0,world2);
        }
    }

   // generating the new world
    for (x = xstart; x < xend; x++){
        for (y = 0; y < N; y++){
            neighbors(x,y,world1,&nn,&n1,&n2);
            cell=read_cell(x,y,0,0,world1);
            if(nn<2||nn>3){
                nextvalue=0;
            }else{
                if(cell==0&&nn==3){//Reproduction si cell=0 et neighbors = 3
                    nextvalue=n1>n2?1:2;
                }else{//Sinon survie
                    nextvalue=cell;
                }
            }
            if(nextvalue!=cell){
                change=1;
            }
            write_cell(x,y,nextvalue,world2);
        }
    }
    return change;
}

// cleaning the screen
void cls(){
    for (int i = 0; i < 10; i++){
        fprintf(stdout,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    }
}

// diplaying the world
void print(unsigned int *world){
    int i;
    cls();
    for (i = 0; i < N; i++)  fprintf(stdout,"−−");
    for (i = 0; i < N*N; i++){
        if (i%N == 0)  fprintf(stdout,"\n");
        if (world[i] == 0)  fprintf(stdout,"  ");
        if (world[i] == 1)  fprintf(stdout,"o ");
        if (world[i] == 2)  fprintf(stdout,"x ");
    }
    fprintf(stdout,"\n");

    for (i = 0; i < N; i++)  fprintf(stdout,"−−");
    fprintf(stdout,"\n");
    sleep(1);
}


// main
int main(int argc,char *argv[]){
    int it,change;
    unsigned int *world1,*world2;
    unsigned int *worldaux;

    // getting started
    //world1 = initialize_dummy();
    //world1 = initialize_random();
    world1 = initialize_glider();
    //world1 = initialize_small_exploder();
    world2 = allocate();
    print(world1);

    it = 0;  change = 1;

    int myrank;
    int nb_processes;
    int step_size;

    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nb_processes);
    if(myrank==0){
        if(N%nb_processes!=0){
            printf("Erreur, taille de tableau %d non divisible par %d processes",N,nb_processes);
            exit(-1);
        }
        printf("nb_processes : %d\n", nb_processes);
        printf("nb_cells : %d \n", N);
        step_size = (int) N/nb_processes;
        //On envoie à tout le monde (bcast semble ne pas envoyer au proccess 2 pour une raison inconnue)
        for (int i = 1; i < nb_processes; i++) {
            MPI_Isend(world1, N*N, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(world2, N*N, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &request);
        }
        // MPI_Bcast(world1, N*N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        // MPI_Bcast(world2, N*N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }else{
        printf("procces %d is a worker\n", myrank);
        MPI_Recv(world1, N*N, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(world2, N*N, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("Initialisation finie\n");

    while (change && it < itMax){
        change = newgeneration(world1,world2,myrank*step_size,(myrank+1)*step_size);
        //Envoi au process précédent (rebouclage si négatif)
        MPI_Isend(world2+myrank*step_size, N*step_size,MPI_UNSIGNED, (myrank+nb_processes-1)%nb_processes,myrank,MPI_COMM_WORLD, &request);
        //Réception du process suivant (rebouclage si négatif)
        MPI_Irecv(world2+myrank*step_size, N*step_size,MPI_UNSIGNED, (myrank+nb_processes+1)%nb_processes,myrank,MPI_COMM_WORLD, &request);

        //Envoi au process suivant (rebouclage si positif)
        MPI_Isend(world2+myrank*step_size, N*step_size,MPI_UNSIGNED, (myrank+nb_processes+1)%nb_processes,myrank,MPI_COMM_WORLD, &request);

        //Réception du process précédent (rebouclage si négatif)
        MPI_Irecv(world2+myrank*step_size, N*step_size,MPI_UNSIGNED, (myrank+nb_processes-1)%nb_processes,myrank,MPI_COMM_WORLD, &request);

        MPI_Barrier(MPI_COMM_WORLD);
        printf("proccess %d après barrière \n",myrank);

        //Envoi au process 0 pour print
        if(myrank!=0){
            MPI_Isend(world2+myrank*step_size, step_size, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &request);
        }else{
            for(int i=1; i<nb_processes; i++){
                MPI_Irecv(world2+i*step_size, step_size, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &request);
            }

            print(world2);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        worldaux = world1;  world1 = world2;  world2 = worldaux;//Switche w1 et w2
        it++;
    }

    MPI_Finalize();


    // ending
    free(world1);   free(world2);
    exit(0);
}
