import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import ForeblocksStudio from './foreblocks-studio.jsx';

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <ForeblocksStudio />
    </StrictMode>,
);
