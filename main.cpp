#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <stdbool.h>

#define CHECKPOINT_FILE "checkpoint_%d.bin"
#define VECTOR_SIZE 10
#define MAX_ITERATIONS 100
#define CHECKPOINT_INTERVAL 5

// Estructura para guardar el estado del proceso
typedef struct {
    int iteration;
    double vector[VECTOR_SIZE];
    int process_rank;
} ProcessState;

// Función para guardar checkpoint
void save_checkpoint(ProcessState state) {
    char filename[50];
    sprintf(filename, CHECKPOINT_FILE, state.process_rank);

    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error al abrir archivo de checkpoint");
        return;
    }

    fwrite(&state, sizeof(ProcessState), 1, file);
    fclose(file);

    printf("Proceso %d: Checkpoint guardado en iteración %d\n",
           state.process_rank, state.iteration);
}

// Función para cargar checkpoint
bool load_checkpoint(int rank, ProcessState* state) {
    char filename[50];
    sprintf(filename, CHECKPOINT_FILE, rank);

    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        return false;
    }

    fread(state, sizeof(ProcessState), 1, file);
    fclose(file);

    printf("Proceso %d: Checkpoint recuperado en iteración %d\n",
           state->process_rank, state->iteration);

    return true;
}

// Función para simular fallo
void simulate_failure(int iteration, int rank) {
    if (iteration == 15 && rank == 0) { // Solo el proceso 0 falla en iteración 15
        printf("Proceso %d: Simulando fallo en iteración %d\n", rank, iteration);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    ProcessState state;
    bool recovered = false;

    // Paso 1: Intentar recuperar checkpoint antes de MPI_Init
    state.process_rank = atoi(getenv("PMI_RANK") ? getenv("PMI_RANK") : "0");
    recovered = load_checkpoint(state.process_rank, &state);

    // Inicialización MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Si no se recuperó checkpoint, inicializar estado
    if (!recovered) {
        state.process_rank = rank;
        state.iteration = 0;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            state.vector[i] = rank * VECTOR_SIZE + i;
        }
        printf("Proceso %d: Iniciando desde cero\n", rank);
    }

    // Bucle principal de computación
    while (state.iteration < MAX_ITERATIONS) {
        // Realizar cómputo (suma de elementos del vector)
        double sum = 0.0;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            state.vector[i] += 0.1;
            sum += state.vector[i];
        }

        printf("Proceso %d: Iteración %d, Suma = %.2f\n",
               rank, state.iteration, sum);

        // Simular fallo en una iteración específica
        simulate_failure(state.iteration, rank);

        // Checkpointing coordinado
        if (state.iteration % CHECKPOINT_INTERVAL == 0) {
            MPI_Barrier(MPI_COMM_WORLD); // Sincronización antes de checkpoint
            save_checkpoint(state);
            MPI_Barrier(MPI_COMM_WORLD); // Sincronización después de checkpoint
        }

        state.iteration++;
    }

    MPI_Finalize();
    return 0;
}
