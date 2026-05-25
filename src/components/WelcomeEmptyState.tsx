'use client';

import {useState, useTransition} from 'react';
import {useRouter} from 'next/navigation';
import {createCourse} from '@/actions/courses';
import {useLanguage} from '@/lib/i18n';
import {NotionIcon} from './NotionIcon';
import {Logo} from './Logo';

interface Props {
  displayName: string;
}

const SAMPLE_COURSES = ['Biology 101', 'CS Theory', 'Anatomy'];

export function WelcomeEmptyState({displayName}: Props) {
  const router = useRouter();
  const {t} = useLanguage();
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const steps = [
    {icon: 'ni-file-upload', title: t('welcome.step1Title'), body: t('welcome.step1Body')},
    {icon: 'ni-comment', title: t('welcome.step2Title'), body: t('welcome.step2Body')},
    {icon: 'ni-rocket', title: t('welcome.step3Title'), body: t('welcome.step3Body')},
  ];

  const handleCreate = (rawName: string) => {
    const trimmed = rawName.trim();
    if (!trimmed) {
      setError(t('welcome.errorEmpty'));
      return;
    }
    setError(null);
    startTransition(async () => {
      const {id, error: serverError} = await createCourse(trimmed);
      if (serverError || !id) {
        setError(serverError ?? t('welcome.errorGeneric'));
        return;
      }
      router.push(`/app?courseId=${id}`);
      router.refresh();
    });
  };

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col items-center px-4 pt-10 pb-16 md:pt-20">
      <div className="w-full rounded-2xl border border-border bg-white p-8 text-center shadow-sm md:p-12">
        <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-50">
          <Logo variant="mark" className="h-9 w-9" />
        </div>

        <h1 className="text-2xl font-semibold tracking-tight text-slate-900 md:text-3xl">
          {t('welcome.headline', {name: displayName})}
        </h1>
        <p className="mx-auto mt-3 max-w-md text-sm text-muted-foreground md:text-base">
          {t('welcome.subtitle')}
        </p>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleCreate(name);
          }}
          className="mx-auto mt-7 flex w-full max-w-md flex-col gap-2 sm:flex-row"
        >
          <label htmlFor="course-name" className="sr-only">
            {t('welcome.create')}
          </label>
          <input
            id="course-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={t('welcome.placeholder')}
            disabled={isPending}
            className="h-11 flex-1 rounded-lg border border-border bg-white px-4 text-sm text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-200 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={isPending}
            className="inline-flex h-11 cursor-pointer items-center justify-center rounded-lg bg-slate-900 px-5 text-sm font-medium text-white transition-colors duration-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isPending ? t('welcome.creating') : t('welcome.create')}
          </button>
        </form>

        {error && (
          <p className="mt-3 text-sm text-rose-600" role="alert">
            {error}
          </p>
        )}

        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-muted-foreground">{t('welcome.orTry')}</span>
          {SAMPLE_COURSES.map((sample) => (
            <button
              key={sample}
              type="button"
              onClick={() => {
                setName(sample);
                handleCreate(sample);
              }}
              disabled={isPending}
              className="cursor-pointer rounded-full border border-border bg-white px-3 py-1 text-xs text-slate-600 transition-colors duration-200 hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {sample}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-10 grid w-full grid-cols-1 gap-4 md:grid-cols-3">
        {steps.map((step, i) => (
          <div
            key={step.title}
            className="rounded-xl border border-border bg-white p-5 text-left shadow-sm"
          >
            <div className="mb-3 flex items-center gap-2">
              <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-600">
                {i + 1}
              </span>
              <NotionIcon name={step.icon} className="h-4 w-4 text-slate-500" />
            </div>
            <h3 className="text-sm font-medium text-slate-900">{step.title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              {step.body}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
