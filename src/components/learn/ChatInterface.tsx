'use client';

import {useEffect, useState, useRef} from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {ChevronDown, FileText, Send, Sparkles, Plus, User, MessageSquare, Trash2, Edit2} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {SourceCard} from '@/components/learn/SourceCard';
import type {Course} from '@/lib/data';
import type {ChatResponse, ChatSource, StoredChatSession} from '@/types/chat';
import {NotionIcon} from '@/components/NotionIcon';
import {deleteChatSession, renameChatSession} from '@/actions/chat';

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
};

export function ChatInterface({
  course,
  initialPdfId,
  profile,
}: {
  course: Course;
  initialPdfId?: string;
  profile?: {
    display_name: string | null;
    avatar_name: string | null;
    avatar_url: string | null;
  } | null;
}) {
  const displayName = profile?.display_name || 'You';
  const avatarName = profile?.avatar_name || 'ni-avatar-male-2';
  const avatarUrl = profile?.avatar_url ?? null;
  const allPdfs = [
    ...course.loose_pdfs,
    ...course.folders.flatMap((folder) => folder.pdfs),
  ];
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<StoredChatSession[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [selectedPdfIds, setSelectedPdfIds] = useState<string[]>(
    initialPdfId && allPdfs.some((pdf) => pdf.id === initialPdfId) 
      ? [initialPdfId] 
      : allPdfs.map(p => p.id),
  );
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    async function loadSessions() {
      const res = await fetch(`/api/chat?courseId=${course.id}`);
      if (!res.ok) {
        return;
      }
      const data = await res.json();
      setSessions(data.sessions ?? []);
    }
    loadSessions();
  }, [course.id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  function togglePdf(pdfId: string) {
    setSelectedPdfIds((current) =>
      current.includes(pdfId)
        ? current.filter((id) => id !== pdfId)
        : [...current, pdfId],
    );
  }

  function startNewChat() {
    setSessionId(undefined);
    setMessages([]);
    setError(null);
  }

  async function onSubmit(event?: React.FormEvent<HTMLFormElement>) {
    if (event) event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed || isLoading) {
      return;
    }
    setIsLoading(true);
    setError(null);
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmed,
    };
    setMessages((current) => [...current, userMessage]);
    setMessage(''); // Clear input early for better UX

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          message: trimmed,
          use_rag: true,
          source_types: ['pdf', 'note', 'annotation_comment'],
          top_k: 8,
          course_id: course.id,
          session_id: sessionId,
          pdf_ids: selectedPdfIds.length === allPdfs.length ? [] : selectedPdfIds,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error ?? 'Chat request failed.');
      }
      const chatResponse = data as ChatResponse;
      setSessionId(chatResponse.session_id);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: chatResponse.answer,
          sources: chatResponse.sources,
        },
      ]);
      setSessions((current) => {
        const existing = current.filter((session) => session.id !== chatResponse.session_id);
        const existingSession = current.find((session) => session.id === chatResponse.session_id);
        return [
          {
            id: chatResponse.session_id ?? sessionId ?? crypto.randomUUID(),
            title: existingSession?.title ?? trimmed.slice(0, 80),
            course_id: course.id,
            updated_at: new Date().toISOString(),
            messages: [
              ...(existingSession?.messages ?? []),
              {
                id: userMessage.id,
                role: 'user',
                content: trimmed,
                sources: [],
                pdf_ids: selectedPdfIds,
                created_at: new Date().toISOString(),
              },
              {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: chatResponse.answer,
                sources: chatResponse.sources,
                pdf_ids: selectedPdfIds,
                created_at: new Date().toISOString(),
              }
            ],
          },
          ...existing,
        ];
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Chat request failed.');
      setMessage(trimmed); // Restore message on failure
    } finally {
      setIsLoading(false);
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="flex flex-1 w-full h-full bg-background overflow-hidden relative">
      {/* Left Sidebar */}
      <div className="hidden md:flex w-[280px] shrink-0 border-r border-border bg-gray-50/30 flex-col h-full">
        <div className="p-5 flex items-center justify-between border-b border-border/50">
          <div className="font-semibold text-sm flex items-center gap-2">
            <NotionIcon name="ni-award" className="w-[20px] h-[20px]" />
            Learn & Research
          </div>
          <button onClick={startNewChat} className="text-muted-foreground hover:text-foreground transition-colors" title="New Chat">
            <Plus className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-8 scrollbar-thin">
          {/* Retrieval Files */}
          <div>
             <div className="mb-3 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Course Materials</div>
             <div className="space-y-1">
               <label className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted/80 transition-colors">
                  <input
                    type="checkbox"
                    className="rounded border-border text-foreground focus:ring-foreground accent-foreground"
                    checked={selectedPdfIds.length === allPdfs.length && allPdfs.length > 0}
                    onChange={(e) => setSelectedPdfIds(e.target.checked ? allPdfs.map(p => p.id) : [])}
                  />
                  <span className="min-w-0 truncate text-muted-foreground font-medium">Use all materials</span>
               </label>
               {allPdfs.map((pdf) => (
                 <label
                   key={pdf.id}
                   className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted/80 transition-colors"
                 >
                   <input
                     type="checkbox"
                     className="rounded border-border text-foreground focus:ring-foreground accent-foreground"
                     checked={selectedPdfIds.includes(pdf.id)}
                     onChange={() => togglePdf(pdf.id)}
                   />
                   <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                   <span className="min-w-0 truncate">{pdf.name}</span>
                 </label>
               ))}
             </div>
          </div>

          {/* Chats */}
          {sessions.length > 0 && (
            <div>
               <div className="mb-3 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Recent Chats</div>
               <div className="space-y-1">
                 {sessions.map((session) => (
                   <div
                     key={session.id}
                     className={`group w-full flex items-center justify-between gap-1.5 rounded-md px-2 py-1.5 text-left text-sm transition-colors ${sessionId === session.id ? 'bg-black/5 text-foreground font-medium' : 'text-muted-foreground hover:bg-black/5 hover:text-foreground'}`}
                   >
                     {editingSessionId === session.id ? (
                       <form 
                         className="flex flex-1 items-center gap-2"
                         onSubmit={async (e) => {
                           e.preventDefault();
                           try {
                             await renameChatSession(session.id, editingTitle);
                             setSessions((current) => current.map(s => s.id === session.id ? { ...s, title: editingTitle || 'Untitled chat' } : s));
                             setEditingSessionId(null);
                           } catch (err) {
                             console.error('Failed to rename', err);
                           }
                         }}
                       >
                         <input 
                           autoFocus
                           className="flex-1 bg-white border border-border rounded px-1.5 py-0.5 text-xs text-foreground outline-none w-full"
                           value={editingTitle}
                           onChange={(e) => setEditingTitle(e.target.value)}
                           onBlur={() => {
                             // small timeout to allow form submission to happen first if it was a button click
                             setTimeout(() => {
                               if (editingSessionId === session.id) setEditingSessionId(null);
                             }, 100);
                           }}
                         />
                       </form>
                     ) : (
                       <>
                         <button
                           type="button"
                           onClick={() => {
                             setSessionId(session.id);
                             setMessages(session.messages.map((stored) => ({
                               id: stored.id,
                               role: stored.role,
                               content: stored.content,
                               sources: stored.sources,
                             })));
                           }}
                           className="flex flex-1 items-center gap-2.5 overflow-hidden"
                         >
                           <MessageSquare className="h-[14px] w-[14px] shrink-0" />
                           <span className="truncate">{session.title ?? 'Untitled chat'}</span>
                         </button>
                         <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                           <button 
                             onClick={(e) => {
                               e.stopPropagation();
                               setEditingSessionId(session.id);
                               setEditingTitle(session.title ?? 'Untitled chat');
                             }}
                             className="p-1 hover:bg-black/10 rounded text-muted-foreground hover:text-foreground"
                           >
                             <Edit2 className="h-3 w-3" />
                           </button>
                           <button 
                             onClick={async (e) => {
                               e.stopPropagation();
                               if (confirm('Delete this chat?')) {
                                 try {
                                   await deleteChatSession(session.id);
                                   setSessions((current) => current.filter(s => s.id !== session.id));
                                   if (sessionId === session.id) startNewChat();
                                 } catch (err) {
                                   console.error('Failed to delete', err);
                                 }
                               }
                             }}
                             className="p-1 hover:bg-red-500/10 rounded text-muted-foreground hover:text-red-500"
                           >
                             <Trash2 className="h-3 w-3" />
                           </button>
                         </div>
                       </>
                     )}
                   </div>
                 ))}
               </div>
            </div>
          )}
        </div>
      </div>

      {/* Right Area */}
      <div className="flex-1 flex flex-col h-full bg-background relative overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto w-full px-4 md:px-8 pt-8 pb-40">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-4 animate-in fade-in duration-700">
                <Sparkles className="w-8 h-8 text-muted-foreground/40 mb-2" />
                <h2 className="text-xl font-medium tracking-tight">How can I help you learn?</h2>
                <p className="text-muted-foreground max-w-sm mx-auto text-sm">
                  Ask questions across all your course materials, or select specific files to narrow the search.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((chatMessage) => (
                  <div
                    key={chatMessage.id}
                    className={`flex gap-4 p-5 rounded-2xl group animate-in fade-in duration-300 ${chatMessage.role === 'user' ? 'bg-white border border-border/60 shadow-sm' : 'bg-[#F7F7F5] border border-transparent'}`}
                  >
                    <div className="mt-1 shrink-0 flex items-center justify-center">
                      {chatMessage.role === 'user' ? (
                        <div className="h-6 w-6 rounded-md bg-white shadow-sm border border-border overflow-hidden flex items-center justify-center">
                          {avatarUrl ? (
                            <img src={avatarUrl} alt={displayName} className="w-full h-full object-cover" />
                          ) : (
                            <div className="p-0.5 w-full h-full flex items-center justify-center">
                              <NotionIcon name={avatarName} className="w-full h-full" />
                            </div>
                          )}
                        </div>
                      ) : (
                        <Sparkles className="h-5 w-5 text-black" />
                      )}
                    </div>
                    <div className="flex-1 space-y-1.5 overflow-hidden">
                      <div className="font-semibold text-sm">
                        {chatMessage.role === 'user' ? displayName : 'Learncycle'}
                      </div>
                      {chatMessage.role === 'assistant' ? (
                        <div className="prose prose-sm md:prose-base dark:prose-invert max-w-none break-words text-foreground leading-relaxed">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {chatMessage.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <div className="prose prose-sm md:prose-base dark:prose-invert max-w-none break-words text-foreground leading-relaxed whitespace-pre-wrap">
                          {chatMessage.content}
                        </div>
                      )}
                      {chatMessage.sources && chatMessage.sources.length > 0 && (
                        <SourceReferences sources={chatMessage.sources} />
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex gap-4 p-5 rounded-2xl bg-[#F7F7F5] border border-transparent animate-pulse">
                    <div className="mt-1 shrink-0 flex items-center justify-center">
                      <Sparkles className="h-5 w-5 text-black/50" />
                    </div>
                    <div className="flex-1 space-y-1.5">
                      <div className="font-semibold text-sm text-black/50">Learncycle</div>
                      <div className="text-muted-foreground text-sm flex gap-1 items-center">
                        Thinking <span className="flex gap-0.5"><span className="animate-bounce">.</span><span className="animate-bounce delay-75">.</span><span className="animate-bounce delay-150">.</span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} className="h-4" />
              </div>
            )}
            
            {error && (
              <div className="mt-6 rounded-lg border border-red-200 bg-red-50/50 px-4 py-3 text-sm text-red-700 flex items-start gap-3">
                 <div className="shrink-0 mt-0.5">⚠️</div>
                 <div>{error}</div>
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 inset-x-0 bg-white pt-6 pb-6 px-4 md:px-8 border-t border-border/40 pointer-events-none">
          <div className="max-w-3xl mx-auto w-full pointer-events-auto">
            <form onSubmit={onSubmit} className="relative flex items-end gap-2 bg-white border border-border shadow-sm rounded-xl px-3 py-2 focus-within:border-black/30 transition-all">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={onKeyDown}
                maxLength={2000}
                rows={1}
                className="max-h-[200px] flex-1 resize-none bg-transparent py-1.5 px-1 text-sm outline-none placeholder:text-muted-foreground/70"
                placeholder="Ask about your materials..."
                style={{ minHeight: '32px' }}
              />
              <button 
                type="submit" 
                disabled={isLoading || !message.trim() || selectedPdfIds.length === 0} 
                className="h-8 w-8 shrink-0 flex items-center justify-center rounded-lg bg-black hover:bg-black/80 transition-all mb-0.5 disabled:opacity-30 disabled:hover:bg-black" 
                title="Send"
              >
                <Send className="h-3.5 w-3.5 text-white" />
              </button>
            </form>
            <div className="text-center mt-2 text-[11px] text-muted-foreground/60">
              AI can make mistakes. Always check course materials for accuracy.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SourceReferences({sources}: {sources: ChatSource[]}) {
  const [open, setOpen] = useState(false);
  const visibleSources = sources.slice(0, 3);
  const hiddenCount = Math.max(0, sources.length - visibleSources.length);

  return (
    <div className="mt-4 pt-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {visibleSources.map((source) => (
          <SourceChip key={source.chunk_id} source={source} />
        ))}
        {hiddenCount > 0 && (
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="flex h-7 items-center gap-1 rounded-md bg-muted/40 px-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted/80 hover:text-foreground"
          >
            {open ? 'Show less' : `+${hiddenCount} more`}
            <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
          </button>
        )}
        {hiddenCount === 0 && (
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="flex h-7 items-center gap-1 rounded-md bg-muted/40 px-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted/80 hover:text-foreground"
          >
            {open ? 'Hide details' : 'Source details'}
            <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
          </button>
        )}
      </div>
      {open && (
        <div className="mt-3 grid gap-2">
          {sources.map((source) => (
            <SourceCard key={source.chunk_id} source={source} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceChip({source}: {source: ChatSource}) {
  const title = source.title
    ?? source.metadata.filename
    ?? (source.source_type === 'pdf'
      ? 'PDF'
      : source.source_type === 'chat_memory'
        ? 'Chat Memory'
      : source.source_type === 'knowledge_graph'
        ? 'Knowledge Graph'
      : source.source_type === 'annotation_comment'
        ? 'Annotation'
        : source.source_type === 'note'
          ? 'Note'
          : 'Source');
  const page = source.page !== null ? `p. ${source.page}` : null;
  const sourceType = source.source_type === 'chat_memory'
    ? 'memory'
    : source.source_type === 'knowledge_graph'
      ? 'graph'
      : source.source_type.replace('_', ' ');

  return (
    <span className="inline-flex h-7 max-w-full items-center gap-1.5 rounded-md border border-border/60 bg-white px-2 text-xs text-muted-foreground shadow-sm">
      <FileText className="h-3 w-3 shrink-0" />
      <span className="shrink-0 rounded bg-muted px-1 py-0.5 text-[10px] font-semibold uppercase leading-none text-foreground">
        {sourceType}
      </span>
      <span className="min-w-0 max-w-[180px] truncate text-foreground">{title}</span>
      {page && <span className="shrink-0 text-muted-foreground">{page}</span>}
    </span>
  );
}
