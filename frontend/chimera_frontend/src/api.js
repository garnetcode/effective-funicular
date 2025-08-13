import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const getNetworks = () => client.get('/networks/');

export const createNetwork = (config) => client.post('/networks/', config);

export const getNetworkStructure = (networkId) => client.get(`/networks/${networkId}/structure/`);

export const learnText = (networkId, text) => client.post(`/networks/${networkId}/learn_text/`, { text });

export const organizeNetwork = (networkId, cue_text = null) => client.post(`/networks/${networkId}/organize/`, { cue_text });
