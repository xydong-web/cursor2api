import { timingSafeEqual } from 'crypto';
import type { NextFunction, Request, Response } from 'express';
import { getConfig } from './config.js';

function isOpenAIStylePath(path: string): boolean {
    return path === '/v1/chat/completions'
        || path === '/chat/completions'
        || path === '/v1/responses'
        || path === '/responses'
        || path === '/v1/models';
}

function extractApiKey(req: Request): string | undefined {
    const xApiKey = req.header('x-api-key')?.trim();
    if (xApiKey) return xApiKey;

    const authHeader = req.header('authorization')?.trim();
    if (!authHeader) return undefined;

    const match = authHeader.match(/^Bearer\s+(.+)$/i);
    return match?.[1]?.trim();
}

function isValidApiKey(providedKey: string | undefined, expectedKey: string): boolean {
    if (!expectedKey) return true;
    if (!providedKey) return false;

    const provided = Buffer.from(providedKey);
    const expected = Buffer.from(expectedKey);

    if (provided.length !== expected.length) return false;
    return timingSafeEqual(provided, expected);
}

function sendUnauthorized(req: Request, res: Response): void {
    const message = 'Invalid API key. Please use x-api-key or Authorization: Bearer <api-key>.';
    res.setHeader('WWW-Authenticate', 'Bearer realm="cursor2api"');

    if (isOpenAIStylePath(req.path)) {
        res.status(401).json({
            error: {
                message,
                type: 'invalid_request_error',
                code: 'invalid_api_key',
            },
        });
        return;
    }

    if (req.path === '/v1/messages' || req.path === '/messages' || req.path === '/v1/messages/count_tokens' || req.path === '/messages/count_tokens') {
        res.status(401).json({
            type: 'error',
            error: {
                type: 'authentication_error',
                message,
            },
        });
        return;
    }

    res.status(401).json({
        error: 'unauthorized',
        message,
    });
}

export function createApiKeyMiddleware() {
    return (req: Request, res: Response, next: NextFunction): void => {
        if (req.method === 'OPTIONS' || req.path === '/health') {
            next();
            return;
        }

        const { apiKey } = getConfig();
        if (!apiKey) {
            next();
            return;
        }

        const providedKey = extractApiKey(req);
        if (isValidApiKey(providedKey, apiKey)) {
            next();
            return;
        }

        sendUnauthorized(req, res);
    };
}