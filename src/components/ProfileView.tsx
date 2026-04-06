'use client';

import {useRef, useState} from 'react';
import {NotionIcon} from './NotionIcon';
import {updateProfile, uploadAvatar} from '@/actions/profile';
import {useRouter} from 'next/navigation';

interface Profile {
  id: string;
  display_name: string | null;
  avatar_name: string | null;
  avatar_url: string | null;
}

const AVATARS = [
  'ni-avatar-male-2',
  'ni-avatar-male-4',
  'ni-avatar-male-8',
  'ni-avatar-female-1',
  'ni-avatar-female-5',
];

export function ProfileView({profile}: {profile: Profile}) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [displayName, setDisplayName] = useState(profile.display_name ?? '');
  const [selectedAvatar, setSelectedAvatar] = useState(profile.avatar_name ?? AVATARS[0]);
  // 'custom' when user has uploaded a photo; 'preset' when using a preset icon
  const [avatarMode, setAvatarMode] = useState<'custom' | 'preset'>(
    profile.avatar_url ? 'custom' : 'preset',
  );
  const [avatarUrl, setAvatarUrl] = useState<string | null>(profile.avatar_url);
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{type: 'success' | 'error'; text: string} | null>(null);

  const handleAvatarFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage(null);
    const fd = new FormData();
    fd.append('avatar', file);
    const result = await uploadAvatar(fd);

    if (result.error) {
      setMessage({type: 'error', text: result.error});
    } else if (result.url) {
      setAvatarUrl(result.url);
      setAvatarMode('custom');
      router.refresh();
    }
    setUploading(false);
    // Reset input so re-selecting same file fires onChange again
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handlePresetClick = (avatar: string) => {
    setSelectedAvatar(avatar);
    setAvatarMode('preset');
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSaving(true);
    setMessage(null);

    const fd = new FormData();
    fd.append('display_name', displayName);
    if (avatarMode === 'preset') {
      fd.append('avatar_name', selectedAvatar);
      fd.append('use_preset', 'true');
    }

    const result = await updateProfile(fd);
    if (result?.error) {
      setMessage({type: 'error', text: 'Failed to update profile.'});
    } else {
      setMessage({type: 'success', text: 'Profile updated successfully!'});
      if (avatarMode === 'preset') setAvatarUrl(null);
      router.refresh();
    }
    setSaving(false);
  };

  return (
    <div className="flex flex-1 flex-col overflow-y-auto bg-white p-8 md:p-16">
      <div className="max-w-2xl mx-auto w-full">
        <header className="mb-12">
          <h1 className="text-4xl font-bold tracking-tight text-foreground mb-4">Account Settings</h1>
          <p className="text-muted-foreground text-lg">Manage your profile and appearance.</p>
        </header>

        <form onSubmit={handleSubmit} className="space-y-12">
          {/* Current avatar + upload */}
          <section className="space-y-6">
            <h2 className="text-xl font-semibold border-b border-border pb-3">Profile Picture</h2>

            <div className="flex items-center gap-6">
              {/* Current avatar preview */}
              <div className="relative shrink-0">
                {avatarMode === 'custom' && avatarUrl ? (
                  <img
                    src={avatarUrl}
                    alt="Profile"
                    className="w-20 h-20 rounded-2xl object-cover border border-border shadow-sm"
                  />
                ) : (
                  <div className="w-20 h-20 rounded-2xl bg-gray-100 border border-border shadow-sm flex items-center justify-center p-2">
                    <NotionIcon name={selectedAvatar} className="w-full h-full" />
                  </div>
                )}
                {uploading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-white/70 rounded-2xl">
                    <div className="w-6 h-6 border-2 border-black/20 border-t-black rounded-full animate-spin" />
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-2">
                <button
                  type="button"
                  disabled={uploading}
                  onClick={() => fileInputRef.current?.click()}
                  className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-black/5 transition-colors disabled:opacity-50"
                >
                  {uploading ? 'Uploading…' : 'Upload photo'}
                </button>
                <p className="text-xs text-muted-foreground">JPG, PNG or GIF up to 5 MB</p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleAvatarFileChange}
              />
            </div>

            {/* Preset avatars */}
            <div>
              <p className="text-sm text-muted-foreground mb-4">Or choose a preset avatar:</p>
              <div className="flex flex-wrap gap-4 items-center">
                {AVATARS.map((avatar) => {
                  const active = avatarMode === 'preset' && selectedAvatar === avatar;
                  return (
                    <button
                      key={avatar}
                      type="button"
                      onClick={() => handlePresetClick(avatar)}
                      className={`relative p-3 rounded-2xl border-2 transition-all hover:bg-black/5 ${
                        active
                          ? 'border-black bg-black/5 scale-110 shadow-md'
                          : 'border-transparent'
                      }`}
                    >
                      <NotionIcon name={avatar} className="w-[56px] h-[56px]" />
                      {active && (
                        <div className="absolute -top-2 -right-2 bg-black text-white rounded-full p-1 border-2 border-white">
                          <NotionIcon name="ni-check" className="w-[10px] h-[10px]" />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
              <a
                href="https://faces.notion.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-5 inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-black/5 transition-colors"
              >
                Create your own Notion avatar
                <NotionIcon name="ni-arrow-up-right" className="w-[16px] h-[16px]" />
              </a>
            </div>
          </section>

          {/* Display name */}
          <section className="space-y-6">
            <h2 className="text-xl font-semibold border-b border-border pb-3">Identity</h2>
            <div className="space-y-2">
              <label htmlFor="display_name" className="text-sm font-medium text-muted-foreground">
                Display Name
              </label>
              <input
                id="display_name"
                name="display_name"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Your name"
                className="w-full rounded-xl border border-border px-5 py-4 text-lg outline-none focus:border-black transition-all bg-gray-50/50"
                required
              />
            </div>
          </section>

          {message && (
            <div
              className={`p-4 rounded-xl text-center font-medium ${
                message.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
              }`}
            >
              {message.text}
            </div>
          )}

          <div className="pt-6 border-t border-border">
            <button
              type="submit"
              disabled={saving}
              className="button-notion button-notion-primary px-8 py-4 text-lg flex items-center justify-center gap-3 min-w-[180px]"
            >
              {saving ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Saving…
                </>
              ) : (
                'Save Changes'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
