import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const getAgents = () => client.get('/agents/');

export const createAgent = (config) => client.post('/agents/', config);

export const getAgentStructure = (agentId) => client.get(`/agents/${agentId}/structure/`);

export const learnWithAgent = (agentId, text) => {
    // This now uses the generic 'learn' endpoint.
    // The backend ChimeraAgent needs a cortex with this ID.
    return client.post(`/agents/${agentId}/learn/`, {
        cortex_id: 'text_input',
        raw_input: text
    });
};
