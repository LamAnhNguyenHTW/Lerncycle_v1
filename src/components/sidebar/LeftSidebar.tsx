'use client';

import {useState, useEffect} from 'react';
import {CourseSwitcher} from './CourseSwitcher';
import {Course} from '@/lib/data';
import {signOut} from '@/actions/auth';
import {NotionIcon} from '../NotionIcon';
import {useRouter} from 'next/navigation';
import {cn} from '@/lib/utils';
import Link from 'next/link';
import {Globe} from 'lucide-react';
import {useLanguage} from '@/lib/i18n';

interface Props {
  courses: Course[];
  activeCourseId?: string;
  activeTab?: string;
  activePdfId?: string;
  profile: {
    display_name: string | null;
    avatar_name: string | null;
    avatar_url: string | null;
  } | null;
}

export function LeftSidebar({courses, activeCourseId, activeTab = 'home', activePdfId, profile}: Props) {
  const router = useRouter();
  const {language, setLanguage, t} = useLanguage();
  const [collapsed, setCollapsed] = useState(activeTab === 'notetaking' || activeTab === 'learn');

  useEffect(() => {
    if (activeTab === 'notetaking' || activeTab === 'learn') {
      setCollapsed(true);
    }
  }, [activeTab]);
  const displayName = profile?.display_name || 'User';
  const avatarName = profile?.avatar_name || 'ni-avatar-male-2';
  const avatarUrl = profile?.avatar_url ?? null;

  return (
    <aside
      className={cn(
        'relative flex h-full shrink-0 flex-col justify-between border-r border-border bg-[#F7F7F5] py-8 overflow-y-auto transition-[width,padding] duration-200',
        collapsed ? 'w-20 px-3' : 'w-72 px-5',
      )}
    >
      <div className="space-y-8">
        <div className={cn('flex items-center', collapsed ? 'justify-center' : 'justify-between')}>
          {!collapsed && (
            <div className="px-2 text-lg font-semibold tracking-wide text-foreground">
              Learncycle
            </div>
          )}
          <button
            onClick={() => setCollapsed((prev) => !prev)}
            className="rounded-lg p-2 text-muted-foreground hover:bg-black/5 hover:text-foreground"
            title={collapsed ? t('nav.openSidebar') : t('nav.collapseSidebar')}
            aria-label={collapsed ? t('nav.openSidebar') : t('nav.collapseSidebar')}
          >
            <NotionIcon
              name={collapsed ? 'ni-chevron-right-circle' : 'ni-chevron-left-circle'}
              className="w-[20px] h-[20px]"
            />
          </button>
        </div>

        <CourseSwitcher courses={courses} activeCourseId={activeCourseId} collapsed={collapsed} />

        <nav className="flex flex-col gap-2 mt-8">
          <NavItem
            icon={<NotionIcon name="ni-browser" className="w-[24px] h-[24px]" />}
            label={t('nav.home')}
            active={activeTab === 'home'}
            href={`/?courseId=${activeCourseId}&tab=home`}
            collapsed={collapsed}
          />
          <NavItem
            icon={<NotionIcon name="ni-pen-line" className="w-[24px] h-[24px]" />}
            label={t('nav.notetaking')}
            active={activeTab === 'notetaking'}
            href={`/?courseId=${activeCourseId}&tab=notetaking${activePdfId ? `&pdfId=${activePdfId}` : ''}`}
            collapsed={collapsed}
          />
          <NavItem
            icon={<NotionIcon name="ni-award" className="w-[24px] h-[24px]" />}
            label={t('nav.learn')}
            active={activeTab === 'learn'}
            href={`/?courseId=${activeCourseId}&tab=learn${activePdfId ? `&pdfId=${activePdfId}` : ''}`}
            collapsed={collapsed}
          />
          <NavItem
            icon={<NotionIcon name="ni-rocket" className="w-[24px] h-[24px]" />}
            label={t('nav.activeLearning')}
            active={activeTab === 'feynman'}
            href={`/?courseId=${activeCourseId}&tab=feynman`}
            collapsed={collapsed}
          />
          <NavItem
            icon={<NotionIcon name="ni-recycle" className="w-[24px] h-[24px]" />}
            label={t('nav.revision')}
            active={activeTab === 'revision'}
            href={`/?courseId=${activeCourseId}&tab=revision`}
            collapsed={collapsed}
          />
        </nav>
      </div>

      <div className={cn(collapsed && 'space-y-2')}>
        <div className={cn('mb-2 flex items-center rounded-xl px-3 py-2', collapsed ? 'justify-center' : 'justify-between')}>
          {!collapsed && <span className="text-xs font-medium text-muted-foreground">{t('language.label')}</span>}
          <button
            type="button"
            onClick={() => setLanguage(language === 'de' ? 'en' : 'de')}
            className="flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs font-semibold text-muted-foreground hover:bg-black/5 hover:text-foreground"
            title={language === 'de' ? t('language.en') : t('language.de')}
            aria-label={t('language.label')}
          >
            <Globe className="h-4 w-4" />
            {!collapsed && <span>{language.toUpperCase()}</span>}
          </button>
        </div>

        <div
          onClick={() => {
            const params = new URLSearchParams(window.location.search);
            params.set('tab', 'profile');
            router.push(`/?${params.toString()}`);
          }}
          className={cn(
            'flex items-center rounded-xl px-3 py-3 transition-colors cursor-pointer group mb-1',
            collapsed ? 'justify-center' : 'justify-between',
            activeTab === 'profile' ? 'bg-white shadow-sm ring-1 ring-black/5' : 'hover:bg-black/5',
          )}
          title={displayName}
        >
          <div className={cn('flex items-center', collapsed ? 'justify-center' : 'gap-3')}>
            <div className="flex size-10 items-center justify-center rounded-xl bg-white shadow-sm border border-border overflow-hidden">
              {avatarUrl ? (
                <img src={avatarUrl} alt={displayName} className="w-full h-full object-cover" />
              ) : (
                <div className="p-1 w-full h-full flex items-center justify-center">
                  <NotionIcon name={avatarName} className="w-full h-full" />
                </div>
              )}
            </div>
            {!collapsed && <span className="text-base font-medium truncate max-w-[120px]">{displayName}</span>}
          </div>
          {!collapsed && (
            <form action={signOut} className="flex">
              <button
                type="submit"
                onClick={(e) => e.stopPropagation()}
                className="flex items-center justify-center text-muted-foreground hover:text-red-500 p-2 rounded-lg hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
                title={t('nav.logout')}
              >
                <NotionIcon name="ni-power-off" className="w-[20px] h-[20px]" />
              </button>
            </form>
          )}
        </div>

        {collapsed && (
          <form action={signOut} className="flex justify-center">
            <button
              type="submit"
              className="flex items-center justify-center text-muted-foreground hover:text-red-500 p-2 rounded-lg hover:bg-red-50 transition-colors"
              title={t('nav.logout')}
              aria-label={t('nav.logout')}
            >
              <NotionIcon name="ni-power-off" className="w-[20px] h-[20px]" />
            </button>
          </form>
        )}
      </div>
    </aside>
  );
}

function NavItem({
  icon,
  label,
  active,
  href,
  collapsed = false,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  href: string;
  collapsed?: boolean;
}) {
  return (
    <Link
      href={href}
      title={label}
      className={`flex items-center rounded-xl px-4 py-3 text-base font-medium transition-colors ${
        collapsed ? 'justify-center' : 'gap-3'
      } ${
        active
          ? 'bg-white text-foreground shadow-sm'
          : 'text-muted-foreground hover:bg-black/5 hover:text-foreground'
      }`}
    >
      {icon}
      {!collapsed && label}
    </Link>
  );
}
