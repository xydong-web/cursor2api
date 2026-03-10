/**
 * 集成测试：模拟长对话，验证上下文压缩流程
 * 
 * 测试场景：30+ 条消息的工具模式对话，验证：
 * 1. 压缩触发条件（消息数 > 30 或字符数 > 60000）
 * 2. few-shot 头部不被压缩
 * 3. 最近 6 条消息保持原文
 * 4. 中间老消息被压缩
 * 5. 消息数量不变（压缩不丢弃）
 * 6. 总字符数显著减少
 */

// 模拟 CursorMessage 类型
interface CursorMessage {
    parts: { type: string; text?: string }[];
    id: string;
    role: string;
}

// 从 converter.ts 复制常量
const FEWSHOT_COUNT = 2;
const MAX_CURSOR_MESSAGES = 30;
const MAX_CONTEXT_CHARS = 60000;
const KEEP_RECENT_MESSAGES = 6;
const COMPRESS_CONTENT_MAX = 200;

// 从 converter.ts 复制 compressMessage
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

// 模拟 converter.ts 中的压缩流程
function applyCompression(messages: CursorMessage[], hasTools: boolean): { compressedCount: number; charsBefore: number; charsAfter: number } {
    let compressedCount = 0;
    const charsBefore = messages.reduce((s, m) => s + m.parts.reduce((a, p) => a + (p.text?.length ?? 0), 0), 0);

    if (hasTools && messages.length > FEWSHOT_COUNT) {
        if (charsBefore > MAX_CONTEXT_CHARS || messages.length > MAX_CURSOR_MESSAGES) {
            const keepRecentCount = Math.min(KEEP_RECENT_MESSAGES, messages.length - FEWSHOT_COUNT);
            const compressBoundary = messages.length - keepRecentCount;

            for (let i = FEWSHOT_COUNT; i < compressBoundary; i++) {
                const original = messages[i].parts.map(p => p.text ?? '').join('');
                const compressed = compressMessage(messages[i].role, original);
                if (compressed.length < original.length) {
                    messages[i] = { ...messages[i], parts: [{ type: 'text', text: compressed }] };
                    compressedCount++;
                }
            }
        }
    }

    const charsAfter = messages.reduce((s, m) => s + m.parts.reduce((a, p) => a + (p.text?.length ?? 0), 0), 0);
    return { compressedCount, charsBefore, charsAfter };
}

// ==================== 构造测试数据 ====================

function buildLongConversation(turnCount: number): CursorMessage[] {
    const messages: CursorMessage[] = [];

    // few-shot 头部 (2 条)
    messages.push({
        parts: [{ type: 'text', text: 'You are a coding assistant. Use tools with ```json action``` format...' }],
        id: 'fs1', role: 'user'
    });
    messages.push({
        parts: [{ type: 'text', text: 'Understood. I\'ll use the structured format for actions.' }],
        id: 'fs2', role: 'assistant'
    });

    // 模拟 N 轮工具交互
    for (let i = 0; i < turnCount; i++) {
        // 用户请求或工具结果
        if (i === 0) {
            messages.push({
                parts: [{ type: 'text', text: `Please read the file src/module${i}.ts and analyze its structure.\n\nRespond with the appropriate action using the structured format.` }],
                id: `u${i}`, role: 'user'
            });
        } else {
            // 工具结果：模拟真实大小的文件内容
            const fileContent = `import { something } from './utils';\n\nexport class Module${i} {\n` +
                Array.from({ length: 50 }, (_, j) => `    public method${j}(): void { /* implementation line ${j} */ }`).join('\n') +
                `\n}\n`;
            messages.push({
                parts: [{ type: 'text', text: `Action output:\n${fileContent}\n\nBased on the output above, continue with the next appropriate action using the structured format.` }],
                id: `u${i}`, role: 'user'
            });
        }

        // 助手的工具调用
        messages.push({
            parts: [{
                type: 'text',
                text: `Let me examine the structure of module${i}.\n\n\`\`\`json action\n{"tool": "read_file", "parameters": {"file_path": "src/module${i}.ts"}}\n\`\`\``
            }],
            id: `a${i}`, role: 'assistant'
        });
    }

    return messages;
}

// ==================== 运行测试 ====================

let passed = 0, failed = 0;
function assert(name: string, condition: boolean, detail?: string) {
    if (condition) { passed++; console.log(`  ✅ ${name}`); }
    else { failed++; console.log(`  ❌ ${name}${detail ? ': ' + detail : ''}`); }
}

console.log('\n=== 场景 1：短对话（不触发压缩）===');
{
    const msgs = buildLongConversation(3); // 2 few-shot + 6 实际 = 8 条
    const originalCount = msgs.length;
    const { compressedCount, charsBefore, charsAfter } = applyCompression(msgs, true);
    assert('短对话不压缩', compressedCount === 0, `compressed=${compressedCount}`);
    assert('消息数不变', msgs.length === originalCount);
    assert('字符数不变', charsBefore === charsAfter);
    console.log(`  📊 ${msgs.length} 条消息, ${charsBefore} chars`);
}

console.log('\n=== 场景 2：长对话（触发按条数压缩）===');
{
    const msgs = buildLongConversation(20); // 2 + 40 = 42 条消息
    const originalCount = msgs.length;
    const { compressedCount, charsBefore, charsAfter } = applyCompression(msgs, true);
    assert('压缩触发', compressedCount > 0, `compressed=${compressedCount}`);
    assert('消息数不变（压缩不丢弃）', msgs.length === originalCount, `${msgs.length} vs ${originalCount}`);
    assert('总字符减少', charsAfter < charsBefore, `${charsAfter} < ${charsBefore}`);
    assert('压缩率 >50%', charsAfter < charsBefore * 0.5, `ratio=${(charsAfter / charsBefore * 100).toFixed(1)}%`);
    console.log(`  📊 ${msgs.length} 条, ${charsBefore} → ${charsAfter} chars (${(100 - charsAfter / charsBefore * 100).toFixed(1)}% 减少)`);

    // 验证 few-shot 头部不被压缩
    assert('few-shot[0] 保持原文', msgs[0].parts[0].text!.includes('You are a coding assistant'));
    assert('few-shot[1] 保持原文', msgs[1].parts[0].text!.includes('Understood'));

    // 验证最近 6 条保持完整
    const lastSix = msgs.slice(-6);
    for (const m of lastSix) {
        const text = m.parts[0].text!;
        const isOriginal = text.includes('Action output:\n') || text.includes('```json action');
        assert(`最近消息保持原文 (role=${m.role}, ${text.length} chars)`, isOriginal || text.length <= COMPRESS_CONTENT_MAX);
    }

    // 验证中间消息被压缩
    const midMsg = msgs[4]; // 第 5 条消息（应在压缩区）
    assert('中间消息已压缩', midMsg.parts[0].text!.length < 300, `len=${midMsg.parts[0].text!.length}`);
}

console.log('\n=== 场景 3：非工具模式（不压缩）===');
{
    const msgs = buildLongConversation(20);
    const { compressedCount } = applyCompression(msgs, false);
    assert('非工具模式不压缩', compressedCount === 0);
}

console.log('\n=== 场景 4：大字符数但少消息（按字符数触发）===');
{
    const msgs: CursorMessage[] = [
        { parts: [{ type: 'text', text: 'few-shot user' }], id: 'fs1', role: 'user' },
        { parts: [{ type: 'text', text: 'few-shot assistant' }], id: 'fs2', role: 'assistant' },
    ];
    // 10 轮，但每条都很大
    for (let i = 0; i < 10; i++) {
        msgs.push({
            parts: [{ type: 'text', text: `Action output:\n${'x'.repeat(8000)}\n\nBased on the output above, continue with the next appropriate action using the structured format.` }],
            id: `u${i}`, role: 'user'
        });
        msgs.push({
            parts: [{ type: 'text', text: `Analysis done.\n\n\`\`\`json action\n{"tool": "write_file", "parameters": {"file_path": "out${i}.ts", "content": "${'y'.repeat(3000)}"}}\n\`\`\`` }],
            id: `a${i}`, role: 'assistant'
        });
    }
    const originalChars = msgs.reduce((s, m) => s + m.parts.reduce((a, p) => a + (p.text?.length ?? 0), 0), 0);
    assert('超过字符阈值', originalChars > MAX_CONTEXT_CHARS, `${originalChars} > ${MAX_CONTEXT_CHARS}`);

    const { compressedCount, charsBefore, charsAfter } = applyCompression(msgs, true);
    assert('按字符数触发压缩', compressedCount > 0);
    assert('字符数大幅减少', charsAfter < charsBefore * 0.5, `${charsAfter} < ${charsBefore * 0.5}`);
    console.log(`  📊 ${msgs.length} 条, ${charsBefore} → ${charsAfter} chars (${(100 - charsAfter / charsBefore * 100).toFixed(1)}% 减少)`);
}

console.log(`\n=== 总结果: ${passed} 通过, ${failed} 失败 ===\n`);
process.exit(failed > 0 ? 1 : 0);
