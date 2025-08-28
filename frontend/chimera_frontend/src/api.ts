import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// --- Type Definitions ---

export interface Environment {
  id: string;
  name: string;
}

export interface GraphData {
  // Define the structure of your graph data here
  nodes: any[];
  edges: any[];
  // Add other properties from the backend response
}

// --- API Functions ---

export const getEnvironments = (): Promise<AxiosResponse<Environment[]>> => client.get('/environments/');

export const getAgentStructure = (agentId: string): Promise<AxiosResponse<GraphData>> => client.get(`/agents/${agentId}/structure/`);

export const learnWithAgent = async (agentId: string, text: string): Promise<AxiosResponse<any>> => {
    // This function now implements the two-step process required by the refactored backend.
    // 1. Learn the pattern associatively.
    console.log(`Step 1: Learning pattern for agent ${agentId}`);
    const learnResponse = await client.post(`/agents/${agentId}/learn_associative/`, {
        cortex_id: 'text_input', // Assuming a default text cortex
        raw_input: text
    });

    const patternId = learnResponse.data.pattern_id;
    if (patternId === undefined) {
        throw new Error("Learning did not return a pattern_id.");
    }

    // 2. Trigger the memory organization step with the new pattern's ID.
    console.log(`Step 2: Organizing memory for pattern_id ${patternId}`);
    const organizeResponse = await client.post(`/agents/${agentId}/organize_memory/`, {
        pattern_id: patternId
    });

    return organizeResponse;
};

export const startTraining = (agentId: string, envId: string): Promise<AxiosResponse<any>> => client.post(`/agents/${agentId}/start_training/`, { env_id: envId });
