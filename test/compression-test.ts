/**
 * 快速测试：上下文压缩 + tolerantParse 增强
 */

// ==================== 1. tolerantParse 测试 ====================

// 内联一个简化版 tolerantParse 进行测试
function tolerantParse(jsonStr: string): any {
    try { return JSON.parse(jsonStr); } catch {}

    let inString = false, escaped = false, fixed = '';
    const bracketStack: string[] = [];
    for (let i = 0; i < jsonStr.length; i++) {
        const char = jsonStr[i];
        if (char === '\\' && !escaped) { escaped = true; fixed += char; }
        else if (char === '"' && !escaped) { inString = !inString; fixed += char; escaped = false; }
        else {
            if (inString) {
                if (char === '\n') fixed += '\\n';
                else if (char === '\r') fixed += '\\r';
                else if (char === '\t') fixed += '\\t';
                else fixed += char;
            } else {
                if (char === '{' || char === '[') bracketStack.push(char === '{' ? '}' : ']');
                else if (char === '}' || char === ']') { if (bracketStack.length > 0) bracketStack.pop(); }
                fixed += char;
            }
            escaped = false;
        }
    }
    if (inString) fixed += '"';
    while (bracketStack.length > 0) fixed += bracketStack.pop();
    fixed = fixed.replace(/,\s*([}\]])/g, '$1');

    try { return JSON.parse(fixed); } catch (_e2) {
        const lastBrace = fixed.lastIndexOf('}');
        if (lastBrace > 0) { try { return JSON.parse(fixed.substring(0, lastBrace + 1)); } catch {} }

        // 第四层：正则兜底
        try {
            const toolMatch = jsonStr.match(/"(?:tool|name)"\s*:\s*"([^"]+)"/);
            if (toolMatch) {
                const toolName = toolMatch[1];
                const paramsMatch = jsonStr.match(/"(?:parameters|arguments|input)"\s*:\s*(\{[\s\S]*)/);
                let params: Record<string, unknown> = {};
                if (paramsMatch) {
                    const paramsStr = paramsMatch[1];
                    let depth = 0, end = -1, pInString = false, pEscaped = false;
                    for (let i = 0; i < paramsStr.length; i++) {
                        const c = paramsStr[i];
                        if (c === '\\' && !pEscaped) { pEscaped = true; continue; }
                        if (c === '"' && !pEscaped) { pInString = !pInString; }
                        if (!pInString) {
                            if (c === '{') depth++;
                            if (c === '}') { depth--; if (depth === 0) { end = i; break; } }
                        }
                        pEscaped = false;
                    }
                    if (end > 0) {
                        const rawParams = paramsStr.substring(0, end + 1);
                        try { params = JSON.parse(rawParams); } catch {
                            const fieldRegex = /"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"/g;
                            let fm;
                            while ((fm = fieldRegex.exec(rawParams)) !== null) {
                                params[fm[1]] = fm[2].replace(/\\n/g, '\n').replace(/\\t/g, '\t');
                            }
                        }
                    }
                }
                return { tool: toolName, parameters: params };
            }
        } catch {}
        throw _e2;
    }
}

// ==================== 2. compressMessage 测试 ====================
const COMPRESS_CONTENT_MAX = 200;

function compressMessage(role: string, text: string): string {
    if (text.length <= COMPRESS_CONTENT_MAX) return text;
    if (role === 'user') {
        const actionMatch = text.match(/^Action output:\n([\s\S]*?)(?:\n\nBased on the output above|$)/);
        if (actionMatch) {
            const output = actionMatch[1];
            const firstLine = output.split('\n')[0]?.trim() || '';
            const lineCount = output.split('\n').length;
            return `Action output: [${output.length} chars, ${lineCount} lines] ${firstLine.substring(0, 80)}...`;
        }
        const errorMatch = text.match(/^The action encountered an error:\n([\s\S]*?)(?:\n\nBased on the output above|$)/);
        if (errorMatch) {
            return `Action error: ${errorMatch[1].substring(0, 150)}...`;
        }
        return text.substring(0, COMPRESS_CONTENT_MAX) + `... [${text.length} chars total]`;
    }
    if (role === 'assistant') {
        const toolBlocks = text.match(/```json action\s*\n([\s\S]*?)```/g);
        if (toolBlocks && toolBlocks.length > 0) {
            const summaries: string[] = [];
            for (const block of toolBlocks) {
                try {
                    const jsonMatch = block.match(/```json action\s*\n([\s\S]*?)```/);
                    if (jsonMatch) {
                        const parsed = JSON.parse(jsonMatch[1]);
                        const toolName = parsed.tool || parsed.name || 'unknown';
                        const paramKeys = parsed.parameters ? Object.keys(parsed.parameters) : [];
                        summaries.push(`[Called ${toolName}(${paramKeys.join(', ')})]`);
                    }
                } catch { summaries.push('[Called action]'); }
            }
            const cleanText = text.replace(/```json action\s*\n[\s\S]*?```/g, '').trim();
            const briefText = cleanText.length > 100 ? cleanText.substring(0, 100) + '...' : cleanText;
            return (briefText ? briefText + '\n' : '') + summaries.join('\n');
        }
        return text.substring(0, COMPRESS_CONTENT_MAX) + `... [${text.length} chars]`;
    }
    return text.substring(0, COMPRESS_CONTENT_MAX) + '...';
}

// ==================== 运行测试 ====================

let passed = 0, failed = 0;
function assert(name: string, condition: boolean, detail?: string) {
    if (condition) { passed++; console.log(`  ✅ ${name}`); }
    else { failed++; console.log(`  ❌ ${name}${detail ? ': ' + detail : ''}`); }
}

console.log('\n=== tolerantParse 测试 ===');

// 正常 JSON
const t1 = tolerantParse('{"tool":"read_file","parameters":{"file_path":"src/index.ts"}}');
assert('正常 JSON', t1.tool === 'read_file' && t1.parameters.file_path === 'src/index.ts');

// 带裸换行符
const t2 = tolerantParse('{"tool":"write_file","parameters":{"content":"line1\nline2"}}');
assert('裸换行修复', t2.tool === 'write_file');

// 截断 JSON（未闭合）
const t3 = tolerantParse('{"tool":"bash","parameters":{"command":"ls -la');
assert('截断兜底', t3.tool === 'bash');

// 含未转义引号的代码内容（最重要的场景）
const badJson = `{
  "tool": "write_file",
  "parameters": {
    "file_path": "test.ts",
    "content": "const x = "hello"; console.log(x);"
  }
}`;
const t4 = tolerantParse(badJson);
assert('未转义引号 - 提取 tool 名', t4.tool === 'write_file');
assert('未转义引号 - 提取参数', Object.keys(t4.parameters).length > 0, `keys=${JSON.stringify(Object.keys(t4.parameters))}`);

// 尾部逗号
const t5 = tolerantParse('{"tool":"list_dir","parameters":{"path":"./",},}');
assert('尾部逗号修复', t5.tool === 'list_dir');

console.log('\n=== compressMessage 测试 ===');

// 短消息不压缩
assert('短消息保留', compressMessage('user', 'hello world') === 'hello world');

// 长工具结果压缩
const longResult = 'Action output:\n' + 'x'.repeat(5000) + '\n\nBased on the output above, continue...';
const c1 = compressMessage('user', longResult);
assert('工具结果压缩', c1.length < 200, `压缩到 ${c1.length} chars`);
assert('工具结果保留信息', c1.includes('5000 chars') && c1.includes('Action output'));

// 错误结果压缩
const errorResult = 'The action encountered an error:\nPermission denied: cannot access /root/secret\n\nBased on the output above, continue...';
const c2 = compressMessage('user', errorResult.padEnd(300, ' detail'));
assert('错误结果压缩', c2.startsWith('Action error:'));

// 助手消息（工具调用）压缩
const assistantMsg = `Let me check the file structure first.\n\n\`\`\`json action\n{"tool":"read_file","parameters":{"file_path":"src/index.ts"}}\n\`\`\`\n\nAnd then more text here to pad the message beyond the threshold limit. ${'x'.repeat(200)}`;
const c3 = compressMessage('assistant', assistantMsg);
assert('助手消息压缩保留工具名', c3.includes('[Called read_file(file_path)]'), c3);
assert('助手消息压缩', c3.length < assistantMsg.length, `${c3.length} < ${assistantMsg.length}`);

// 普通长用户消息
const longUser = 'Please help me with '.padEnd(500, 'this task ');
const c4 = compressMessage('user', longUser);
assert('普通用户消息截短', c4.length < 300 && c4.includes('chars total'));

console.log(`\n=== 结果: ${passed} 通过, ${failed} 失败 ===\n`);
process.exit(failed > 0 ? 1 : 0);
