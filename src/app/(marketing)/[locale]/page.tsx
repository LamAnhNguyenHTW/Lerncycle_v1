import {notFound} from 'next/navigation';
import {isLocale} from '@/lib/locale';
import {getDictionary} from '@/locales';
import {MarketingTopNav} from '@/components/marketing/MarketingTopNav';
import {Hero} from '@/components/marketing/Hero';
import {ProblemSection} from '@/components/marketing/ProblemSection';
import {SolutionSection} from '@/components/marketing/SolutionSection';
import {FeatureGrid} from '@/components/marketing/FeatureGrid';
import {HowItWorks} from '@/components/marketing/HowItWorks';
import {FAQSection} from '@/components/marketing/FAQSection';
import {BetaSignupSection} from '@/components/marketing/BetaSignupSection';
import {MarketingFooter} from '@/components/marketing/MarketingFooter';

interface PageProps {
  params: Promise<{locale: string}>;
}

export default async function LandingPage({params}: PageProps) {
  const {locale} = await params;
  if (!isLocale(locale)) notFound();

  const dict = getDictionary(locale);

  return (
    <div className="flex min-h-screen flex-col overflow-x-clip">
      <MarketingTopNav locale={locale} dict={dict} />
      <main className="flex-1">
        <Hero dict={dict} />
        <ProblemSection dict={dict} />
        <SolutionSection dict={dict} />
        <FeatureGrid dict={dict} />
        <HowItWorks dict={dict} />
        <FAQSection dict={dict} />
        <BetaSignupSection locale={locale} dict={dict} />
      </main>
      <MarketingFooter locale={locale} dict={dict} />
    </div>
  );
}
